//! Tool permission gate: modes, rules, and interactive approval for gated tools.

mod broker;
mod cloud_github;

use std::collections::HashSet;
use std::io::{self, Write};
use std::sync::Arc;

use serde_json::Value;

pub use broker::{
    ApprovalBroker, ApprovalRequest, AutoApprovalBroker, SharedApprovalBroker,
    TerminalApprovalBroker,
};
pub use cloud_github::{deny_cloud_github_cli, deny_git_cli, deny_git_cli_with_message};

/// Shared permission gate used by main agent and subagents.
pub trait ToolPermission: Send + Sync {
    fn approve_tool_call(&mut self, tool_name: &str, arguments: &str) -> io::Result<bool>;
    fn evaluate(&self, tool_name: &str, arguments: &str) -> PolicyDecision;
    fn apply_choice(&mut self, tool_name: &str, arguments: &str, choice: PromptChoice) -> bool;
    fn denied_message(tool_name: &str) -> String
    where
        Self: Sized;
}

pub type SharedToolPermission = Arc<parking_lot::Mutex<dyn ToolPermission>>;

/// Permission modes aligned with grok-style agent safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionMode {
    /// Ask before Write / Edit / Bash / MCP (default).
    #[default]
    Default,
    /// Auto-approve file edits (Write/Edit); still ask for Bash / MCP.
    AcceptEdits,
    /// Never prompt; deny anything that would have required confirmation.
    DontAsk,
    /// Auto-approve all gated tools (alias: yolo / bypassPermissions).
    BypassPermissions,
    /// Legacy aliases kept for config compatibility.
    Manual,
    Yolo,
}

impl PermissionMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "yolo" | "bypass" | "bypasspermissions" | "bypass_permissions" => {
                Self::BypassPermissions
            }
            "accept_edits" | "acceptedits" | "accept-edits" => Self::AcceptEdits,
            "dont_ask" | "dontask" | "dont-ask" => Self::DontAsk,
            "plan" => Self::Default,
            "manual" | "default" | "" => Self::Default,
            _ => Self::Default,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default | Self::Manual => "default",
            Self::AcceptEdits => "accept_edits",
            Self::DontAsk => "dont_ask",
            Self::BypassPermissions | Self::Yolo => "bypass",
        }
    }

    fn auto_approves_all(self) -> bool {
        matches!(self, Self::BypassPermissions | Self::Yolo)
    }

    fn auto_approves_edits(self) -> bool {
        matches!(
            self,
            Self::AcceptEdits | Self::BypassPermissions | Self::Yolo
        )
    }

    fn denies_without_prompt(self) -> bool {
        matches!(self, Self::DontAsk)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptChoice {
    AllowOnce,
    AllowSession,
    Deny,
}

/// Pure policy outcome. `Ask` is the only case that may call [`ApprovalBroker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDecision {
    Allow,
    Deny,
    Ask,
}

/// A single allow/deny/ask rule matched against tool name (glob-ish prefix/suffix `*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRule {
    pub pattern: String,
    pub action: RuleAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleAction {
    Allow,
    Deny,
    Ask,
}

impl PermissionRule {
    pub fn matches(&self, tool_name: &str) -> bool {
        let pat = self.pattern.as_str();
        if pat == "*" {
            return true;
        }
        if let Some(prefix) = pat.strip_suffix('*') {
            return tool_name.starts_with(prefix);
        }
        if let Some(suffix) = pat.strip_prefix('*') {
            return tool_name.ends_with(suffix);
        }
        tool_name == pat
    }
}

pub type PermissionPrompter = dyn Fn(&str, &str) -> io::Result<PromptChoice> + Send + Sync;

/// Gate for Write / Edit / Bash before execution.
pub struct PermissionGate {
    mode: PermissionMode,
    approved_session: HashSet<String>,
    rules: Vec<PermissionRule>,
    prompter: Box<PermissionPrompter>,
    auto_allow_bash: bool,
    deny_git_push: bool,
}

impl PermissionGate {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            approved_session: HashSet::new(),
            rules: Vec::new(),
            prompter: Box::new(default_prompter),
            auto_allow_bash: false,
            deny_git_push: cloud_github::is_cloud_run(),
        }
    }

    pub fn with_prompter(mode: PermissionMode, prompter: Box<PermissionPrompter>) -> Self {
        Self {
            mode,
            approved_session: HashSet::new(),
            rules: Vec::new(),
            prompter,
            auto_allow_bash: false,
            deny_git_push: cloud_github::is_cloud_run(),
        }
    }

    pub fn with_rules(mut self, rules: Vec<PermissionRule>) -> Self {
        self.rules = rules;
        self
    }

    pub fn with_auto_allow_bash(mut self, enabled: bool) -> Self {
        self.auto_allow_bash = enabled;
        self
    }

    pub fn with_deny_git_push(mut self, deny: bool) -> Self {
        self.deny_git_push = deny;
        self
    }

    pub fn set_auto_allow_bash(&mut self, enabled: bool) {
        self.auto_allow_bash = enabled;
    }

    pub fn auto_allow_bash(&self) -> bool {
        self.auto_allow_bash
    }

    pub fn set_deny_git_push(&mut self, deny: bool) {
        self.deny_git_push = deny;
    }

    pub fn deny_git_push(&self) -> bool {
        self.deny_git_push
    }

    pub fn set_rules(&mut self, rules: Vec<PermissionRule>) {
        self.rules = rules;
    }

    pub fn set_prompter(&mut self, prompter: Box<PermissionPrompter>) {
        self.prompter = prompter;
    }

    pub fn inherit_policy_from(&mut self, other: &Self) {
        self.auto_allow_bash = other.auto_allow_bash;
        self.deny_git_push = other.deny_git_push;
        self.rules = other.rules.clone();
        self.approved_session = other.approved_session.clone();
    }

    pub fn mode(&self) -> PermissionMode {
        match self.mode {
            PermissionMode::Manual => PermissionMode::Default,
            PermissionMode::Yolo => PermissionMode::BypassPermissions,
            other => other,
        }
    }

    pub fn requires_confirmation(tool_name: &str) -> bool {
        matches!(tool_name, "Write" | "Edit" | "Bash") || tool_name.starts_with("mcp__")
    }

    fn matching_rule(&self, tool_name: &str) -> Option<&PermissionRule> {
        self.rules.iter().find(|r| r.matches(tool_name))
    }

    /// Check if a tool call is denied by policy configured on this gate.
    pub fn policy_denied(&self, tool_name: &str, arguments: &str) -> Option<String> {
        policy_denied_with_git_push(tool_name, arguments, self.deny_git_push)
    }

    /// Classify a tool call without talking to a user.
    pub fn evaluate(&self, tool_name: &str, arguments: &str) -> PolicyDecision {
        if self.policy_denied(tool_name, arguments).is_some() {
            return PolicyDecision::Deny;
        }

        let forced_ask = matches!(
            self.matching_rule(tool_name).map(|r| r.action),
            Some(RuleAction::Ask)
        );

        if let Some(rule) = self.matching_rule(tool_name) {
            match rule.action {
                RuleAction::Allow => return PolicyDecision::Allow,
                RuleAction::Deny => return PolicyDecision::Deny,
                RuleAction::Ask => {}
            }
        }

        if !forced_ask {
            if self.mode.auto_approves_all() {
                return PolicyDecision::Allow;
            }

            if !Self::requires_confirmation(tool_name) {
                return PolicyDecision::Allow;
            }

            if self.mode.auto_approves_edits() && matches!(tool_name, "Write" | "Edit") {
                return PolicyDecision::Allow;
            }

            if self.auto_allow_bash && tool_name == "Bash" {
                return PolicyDecision::Allow;
            }

            if self.mode.denies_without_prompt() {
                return PolicyDecision::Deny;
            }
        } else if self.mode.denies_without_prompt() {
            return PolicyDecision::Deny;
        }

        if let Some(key) = session_approval_key(tool_name, arguments) {
            if self.approved_session.contains(&key) {
                return PolicyDecision::Allow;
            }
        }

        PolicyDecision::Ask
    }

    /// Record a prompt decision. `AllowSession` is remembered for this gate.
    pub fn apply_choice(&mut self, tool_name: &str, arguments: &str, choice: PromptChoice) -> bool {
        match choice {
            PromptChoice::AllowOnce => true,
            PromptChoice::AllowSession => {
                if let Some(key) = session_approval_key(tool_name, arguments) {
                    self.approved_session.insert(key);
                }
                true
            }
            PromptChoice::Deny => false,
        }
    }

    /// Returns `Ok(true)` if the tool may run, `Ok(false)` if denied.
    pub fn check(&mut self, tool_name: &str, arguments: &str) -> io::Result<bool> {
        match self.evaluate(tool_name, arguments) {
            PolicyDecision::Allow => Ok(true),
            PolicyDecision::Deny => Ok(false),
            PolicyDecision::Ask => {
                let preview = truncate(arguments, 120);
                let choice = (self.prompter)(tool_name, &preview)?;
                Ok(self.apply_choice(tool_name, arguments, choice))
            }
        }
    }

    pub fn denied_message(tool_name: &str) -> String {
        format!("Tool `{tool_name}` was denied by the user.")
    }

    pub fn permission_denied_message(tool_name: &str, arguments: &str) -> String {
        policy_denied(tool_name, arguments).unwrap_or_else(|| Self::denied_message(tool_name))
    }
}

/// Hard deny Write/Edit under protected path segments (aligned with sandbox `.git` rules).
/// Also denies `git push` / `gh` when `deny_git_push` is true.
pub fn policy_denied(tool_name: &str, arguments: &str) -> Option<String> {
    policy_denied_with_git_push(tool_name, arguments, cloud_github::is_cloud_run())
}

pub fn policy_denied_with_git_push(
    tool_name: &str,
    arguments: &str,
    deny_git_push: bool,
) -> Option<String> {
    if deny_git_push && matches!(tool_name, "Bash") {
        if let Some(command) = extract_bash_command(arguments) {
            if let Some(msg) = cloud_github::deny_git_cli(&command) {
                return Some(msg);
            }
        }
    }
    if !matches!(tool_name, "Write" | "Edit") {
        return None;
    }
    let path = extract_tool_path(arguments)?;
    if is_protected_write_path(&path) {
        Some(format!(
            "Policy denied `{tool_name}` on `{path}`: writes under `node_modules/` and `.git/` are blocked."
        ))
    } else {
        None
    }
}

fn extract_bash_command(arguments: &str) -> Option<String> {
    let value: Value = serde_json::from_str(arguments).ok()?;
    value.get("command")?.as_str().map(str::to_string)
}

pub fn approve_tool_call(
    gate: &mut PermissionGate,
    tool_name: &str,
    arguments: &str,
) -> io::Result<bool> {
    gate.check(tool_name, arguments)
}

impl ToolPermission for PermissionGate {
    fn approve_tool_call(&mut self, tool_name: &str, arguments: &str) -> io::Result<bool> {
        self.check(tool_name, arguments)
    }

    fn evaluate(&self, tool_name: &str, arguments: &str) -> PolicyDecision {
        PermissionGate::evaluate(self, tool_name, arguments)
    }

    fn apply_choice(&mut self, tool_name: &str, arguments: &str, choice: PromptChoice) -> bool {
        PermissionGate::apply_choice(self, tool_name, arguments, choice)
    }

    fn denied_message(tool_name: &str) -> String {
        PermissionGate::denied_message(tool_name)
    }
}

/// Resolve a tool call against policy, then the optional async broker.
///
/// The permission mutex is not held while awaiting the broker.
pub async fn resolve_permission(
    permission: &SharedToolPermission,
    broker: Option<&SharedApprovalBroker>,
    request: ApprovalRequest,
) -> io::Result<bool> {
    let decision = permission
        .lock()
        .evaluate(&request.tool_name, &request.arguments);
    match decision {
        PolicyDecision::Allow => Ok(true),
        PolicyDecision::Deny => Ok(false),
        PolicyDecision::Ask => {
            let choice = if let Some(broker) = broker {
                broker.request(request.clone()).await?
            } else {
                return permission
                    .lock()
                    .approve_tool_call(&request.tool_name, &request.arguments);
            };
            Ok(permission
                .lock()
                .apply_choice(&request.tool_name, &request.arguments, choice))
        }
    }
}

fn extract_tool_path(arguments: &str) -> Option<String> {
    let value: Value = serde_json::from_str(arguments).ok()?;
    value
        .get("path")
        .and_then(|p| p.as_str())
        .map(str::to_string)
}

fn is_protected_write_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let trimmed = normalized.trim_start_matches("./");
    path_has_segment(trimmed, "node_modules") || path_has_segment(trimmed, ".git")
}

fn path_has_segment(path: &str, segment: &str) -> bool {
    path.split('/').any(|part| part == segment)
}

pub(crate) fn default_prompter(tool_name: &str, args_preview: &str) -> io::Result<PromptChoice> {
    eprint!("\nAllow {tool_name}({args_preview})? [y]es / [n]o / [a]pprove for session: ");
    let _ = io::stderr().flush();
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    match line.trim().to_lowercase().as_str() {
        "y" | "yes" => Ok(PromptChoice::AllowOnce),
        "a" | "approve" | "always" => Ok(PromptChoice::AllowSession),
        _ => Ok(PromptChoice::Deny),
    }
}

/// Session "approve once" key: `Write:relative/path` or `Bash:command`.
pub fn session_approval_key(tool_name: &str, arguments: &str) -> Option<String> {
    let value: Value = serde_json::from_str(arguments).ok()?;
    match tool_name {
        "Write" | "Edit" => value
            .get("path")
            .and_then(|p| p.as_str())
            .map(|path| format!("{tool_name}:{path}")),
        "Bash" => value
            .get("command")
            .and_then(|c| c.as_str())
            .map(|cmd| format!("Bash:{cmd}")),
        name if name.starts_with("mcp__") => Some(format!("{name}:{arguments}")),
        _ => None,
    }
}

pub(crate) fn truncate(input: &str, max: usize) -> String {
    if input.chars().count() <= max {
        input.to_string()
    } else {
        format!("{}...", input.chars().take(max).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    #[test]
    fn yolo_skips_prompt() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let gate = PermissionGate::with_prompter(PermissionMode::Yolo, {
            Box::new(move |_tool, _args| {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(PromptChoice::Deny)
            })
        });
        let mut gate = gate;
        assert!(gate
            .check("Write", r#"{"path":"a.txt","content":"x"}"#)
            .unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn accept_edits_auto_allows_write_but_asks_bash() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let mut gate = PermissionGate::with_prompter(PermissionMode::AcceptEdits, {
            Box::new(move |_tool, _args| {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(PromptChoice::AllowOnce)
            })
        });
        assert!(gate
            .check("Write", r#"{"path":"a.txt","content":"x"}"#)
            .unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(gate.check("Bash", r#"{"command":"ls"}"#).unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn auto_allow_bash_skips_bash_prompt() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let mut gate = PermissionGate::with_prompter(PermissionMode::Default, {
            Box::new(move |_tool, _args| {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(PromptChoice::Deny)
            })
        })
        .with_auto_allow_bash(true);
        assert!(gate.check("Bash", r#"{"command":"ls"}"#).unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!gate
            .check("Write", r#"{"path":"a.txt","content":"x"}"#)
            .unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dont_ask_denies_gated_tools() {
        let mut gate = PermissionGate::new(PermissionMode::DontAsk);
        assert!(!gate.check("Bash", r#"{"command":"ls"}"#).unwrap());
        assert!(gate.check("Read", r#"{"path":"a.txt"}"#).unwrap());
    }

    #[test]
    fn deny_rule_beats_bypass() {
        let mut gate = PermissionGate::new(PermissionMode::BypassPermissions).with_rules(vec![
            PermissionRule {
                pattern: "Bash".into(),
                action: RuleAction::Deny,
            },
        ]);
        assert!(!gate.check("Bash", r#"{"command":"ls"}"#).unwrap());
        assert!(gate
            .check("Write", r#"{"path":"a.txt","content":"x"}"#)
            .unwrap());
    }

    #[test]
    fn allow_rule_skips_prompt() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let mut gate = PermissionGate::with_prompter(PermissionMode::Default, {
            Box::new(move |_tool, _args| {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(PromptChoice::Deny)
            })
        })
        .with_rules(vec![PermissionRule {
            pattern: "mcp__*".into(),
            action: RuleAction::Allow,
        }]);
        assert!(gate.check("mcp__git__status", r#"{"repo":"."}"#).unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn manual_prompts_and_session_approve() {
        let mut gate = PermissionGate::with_prompter(PermissionMode::Manual, {
            Box::new(|_tool, _args| Ok(PromptChoice::AllowSession))
        });
        let args = r#"{"path":"src/foo.rs","content":"x"}"#;
        assert!(gate.check("Write", args).unwrap());
        assert!(gate.check("Write", args).unwrap());
    }

    #[test]
    fn manual_deny_returns_false() {
        let mut gate = PermissionGate::with_prompter(PermissionMode::Manual, {
            Box::new(|_tool, _args| Ok(PromptChoice::Deny))
        });
        assert!(!gate.check("Bash", r#"{"command":"rm -rf /"}"#).unwrap());
    }

    #[test]
    fn mcp_tools_require_confirmation() {
        assert!(PermissionGate::requires_confirmation("mcp__git__status"));
    }

    #[test]
    fn policy_denies_node_modules_and_git_writes() {
        let mut gate = PermissionGate::new(PermissionMode::Yolo);
        assert!(!gate
            .check(
                "Write",
                r#"{"path":"node_modules/pkg/index.js","content":"x"}"#
            )
            .unwrap());
        assert!(!gate
            .check(
                "Edit",
                r#"{"path":"foo/.git/config","old_string":"a","new_string":"b"}"#
            )
            .unwrap());
        assert!(gate
            .check("Write", r#"{"path":"src/main.rs","content":"x"}"#)
            .unwrap());
    }

    #[test]
    fn policy_denied_message_is_specific() {
        let msg =
            PermissionGate::permission_denied_message("Write", r#"{"path":"node_modules/x"}"#);
        assert!(msg.contains("node_modules"));
    }

    #[test]
    fn git_push_protection_blocks_git_push_and_gh() {
        let push =
            policy_denied_with_git_push("Bash", r#"{"command":"git push origin HEAD"}"#, true);
        assert!(push.unwrap().contains("git push"));
        let gh = policy_denied_with_git_push("Bash", r#"{"command":"gh pr create"}"#, true);
        assert!(gh.unwrap().contains("`gh`"));
        assert!(policy_denied_with_git_push("Bash", r#"{"command":"git status"}"#, true).is_none());
        assert!(policy_denied_with_git_push("Bash", r#"{"command":"git push"}"#, false).is_none());

        let mut gate = PermissionGate::new(PermissionMode::Yolo).with_deny_git_push(true);
        assert_eq!(
            gate.evaluate("Bash", r#"{"command":"git push"}"#),
            PolicyDecision::Deny
        );
        gate.set_deny_git_push(false);
        assert_eq!(
            gate.evaluate("Bash", r#"{"command":"git push"}"#),
            PolicyDecision::Allow
        );
    }

    #[test]
    fn session_key_format() {
        assert_eq!(
            session_approval_key("Write", r#"{"path":"a/b.rs"}"#).as_deref(),
            Some("Write:a/b.rs")
        );
        assert_eq!(
            session_approval_key("Bash", r#"{"command":"ls"}"#).as_deref(),
            Some("Bash:ls")
        );
    }

    #[test]
    fn parse_aliases() {
        assert_eq!(
            PermissionMode::parse("yolo"),
            PermissionMode::BypassPermissions
        );
        assert_eq!(
            PermissionMode::parse("accept_edits"),
            PermissionMode::AcceptEdits
        );
        assert_eq!(PermissionMode::parse("dont_ask"), PermissionMode::DontAsk);
        assert_eq!(PermissionMode::parse("manual"), PermissionMode::Default);
    }

    #[test]
    fn evaluate_is_pure_and_does_not_prompt() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let gate = PermissionGate::with_prompter(PermissionMode::Default, {
            Box::new(move |_tool, _args| {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(PromptChoice::Deny)
            })
        });
        assert_eq!(
            gate.evaluate("Read", r#"{"path":"a.txt"}"#),
            PolicyDecision::Allow
        );
        assert_eq!(
            gate.evaluate("Write", r#"{"path":"a.txt","content":"x"}"#),
            PolicyDecision::Ask
        );
        assert_eq!(
            gate.evaluate("Write", r#"{"path":"node_modules/x","content":"x"}"#),
            PolicyDecision::Deny
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn resolve_permission_uses_broker_for_ask() {
        let permission: SharedToolPermission = Arc::new(parking_lot::Mutex::new(
            PermissionGate::with_prompter(PermissionMode::Default, {
                Box::new(|_tool, _args| panic!("sync prompter should not run"))
            }),
        ));
        let broker: SharedApprovalBroker = Arc::new(AutoApprovalBroker::deny());
        let allowed = resolve_permission(
            &permission,
            Some(&broker),
            ApprovalRequest {
                request_id: "req".into(),
                tool_name: "Bash".into(),
                arguments: r#"{"command":"ls"}"#.into(),
                tool_call_id: None,
            },
        )
        .await
        .unwrap();
        assert!(!allowed);
    }

    #[tokio::test]
    async fn resolve_permission_records_session_approval() {
        let permission: SharedToolPermission = Arc::new(parking_lot::Mutex::new(
            PermissionGate::new(PermissionMode::Default),
        ));
        let broker: SharedApprovalBroker = Arc::new(AutoApprovalBroker {
            choice: PromptChoice::AllowSession,
        });
        let request = ApprovalRequest {
            request_id: "req".into(),
            tool_name: "Write".into(),
            arguments: r#"{"path":"src/foo.rs","content":"x"}"#.into(),
            tool_call_id: None,
        };
        assert!(
            resolve_permission(&permission, Some(&broker), request.clone())
                .await
                .unwrap()
        );
        assert_eq!(
            permission
                .lock()
                .evaluate("Write", r#"{"path":"src/foo.rs","content":"x"}"#),
            PolicyDecision::Allow
        );
    }
}
