//! Optional wiring for [`Agent`](crate::Agent). Default path matches legacy `Agent::new`;
//! inject sandbox/tools/context/MCP when assembling a custom runtime.

use std::sync::Arc;

use anyhow::Result;
use parking_lot::Mutex;
use tracing::{info, warn};
use zene_config::ZeneConfig;
use zene_context::{
    conversation_has_memory_context, memory_reminder_from_store, ContextEngine, FsMemoryStore,
};
use zene_llm::ChatClient;
use zene_mcp::McpManager;
use zene_sandbox::{LocalSandbox, Sandbox};
use zene_session::{AgentRecordWriter, FileSessionStore, SessionRecord, SessionStore};
use zene_tools::{
    default_ask_user_prompter, shared_background_tasks, shared_plan_mode, shared_todo_store_from,
    RuntimeScope, SharedAskUserPrompter, SharedBackgroundTasks, SharedPlanMode, SharedTodoStore,
    ToolRegistry,
};

use crate::plan_mode::{default_plan_approval_prompter, PlanApprovalPrompter};
use crate::tool_dedup::ToolDedup;
use crate::Agent;
use zene_hooks::{ExtensionHook, HookRunner, HookSpec};
use zene_permission::{
    PermissionGate, PermissionMode, PermissionRule, RuleAction, SharedToolPermission,
};
use zene_turn::{FollowUpBuffer, SteerBuffer};
use zene_workspace::{build_system_prompt, FsWorkspaceProvider};

/// How MCP servers are attached when building an [`Agent`].
#[derive(Default)]
enum McpAttach {
    /// Connect from workspace MCP config (same as legacy `Agent::new`).
    #[default]
    Auto,
    /// Do not connect or register MCP tools.
    Skip,
    /// Use a pre-connected manager; `ToolRegistry` must already include its tools.
    Inject(McpManager),
}

/// Fluent builder for [`Agent`]. All fields except the four constructor args use
/// product defaults until overridden.
pub struct AgentBuilder {
    config: ZeneConfig,
    sandbox: LocalSandbox,
    session: SessionRecord,
    permission_mode: PermissionMode,

    client: Option<ChatClient>,
    tools: Option<ToolRegistry>,
    mcp: McpAttach,
    context: Option<ContextEngine>,
    hooks: Option<HookRunner>,
    load_hooks_from_config: bool,
    permission: Option<SharedToolPermission>,
    plan_mode: Option<SharedPlanMode>,
    plan_approval: Option<PlanApprovalPrompter>,
    todos: Option<SharedTodoStore>,
    ask_user: Option<SharedAskUserPrompter>,
    background: Option<SharedBackgroundTasks>,
    record_writer: Option<AgentRecordWriter>,
    session_store: Option<Arc<dyn SessionStore>>,
    model_executor: Option<Arc<dyn zene_model_executor::ModelExecutor>>,
    approval_broker: Option<zene_permission::SharedApprovalBroker>,
    external_session_id: Option<String>,
    include_workspace_context: Option<bool>,
    deny_git_push: Option<bool>,
    extensions: Vec<Arc<dyn ExtensionHook>>,
}

impl AgentBuilder {
    pub fn new(
        config: ZeneConfig,
        sandbox: LocalSandbox,
        session: SessionRecord,
        permission_mode: PermissionMode,
    ) -> Self {
        Self {
            config,
            sandbox,
            session,
            permission_mode,
            client: None,
            tools: None,
            mcp: McpAttach::default(),
            context: None,
            hooks: None,
            load_hooks_from_config: true,
            permission: None,
            plan_mode: None,
            plan_approval: None,
            todos: None,
            ask_user: None,
            background: None,
            record_writer: None,
            session_store: None,
            model_executor: None,
            approval_broker: None,
            external_session_id: None,
            include_workspace_context: None,
            deny_git_push: None,
            extensions: Vec::new(),
        }
    }

    /// Construct a builder initialized with default configuration and runtime primitives for `workdir`.
    pub fn for_workdir(workdir: impl AsRef<std::path::Path>) -> Self {
        let workdir = workdir.as_ref();
        let config = ZeneConfig::load(workdir).unwrap_or_default();
        let sandbox = LocalSandbox::new(workdir);
        let session = SessionRecord::new(workdir);
        let permission_mode = PermissionMode::parse(&config.permission_mode);
        Self::new(config, sandbox, session, permission_mode)
    }

    /// Override the agent configuration.
    pub fn config(mut self, config: ZeneConfig) -> Self {
        self.config = config;
        self
    }

    /// Override the sandbox.
    pub fn sandbox(mut self, sandbox: LocalSandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Override the session record.
    pub fn session(mut self, session: SessionRecord) -> Self {
        self.session = session;
        self
    }

    /// Override the permission mode.
    pub fn permission_mode(mut self, permission_mode: PermissionMode) -> Self {
        self.permission_mode = permission_mode;
        self
    }

    /// Bypass permission checks (convenience for scripts and testing).
    pub fn bypass_permissions(mut self) -> Self {
        self.permission_mode = PermissionMode::BypassPermissions;
        self
    }

    /// Use the minimalist core toolset (read, write, edit, bash, grep, glob).
    pub fn core_tools(mut self) -> Self {
        self.tools = Some(zene_tools::core_tools());
        self
    }

    /// Use the read-only minimal toolset (read, grep, glob).
    pub fn minimal_tools(mut self) -> Self {
        self.tools = Some(zene_tools::minimal_tools());
        self
    }

    /// Use a pre-built chat client instead of `ChatClient::from_config`.
    pub fn client(mut self, client: ChatClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Replace default `RuntimeScope::agent` registry (MCP tools are not merged unless [`Self::mcp_auto`]).
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Skip MCP discovery and registration.
    pub fn without_mcp(mut self) -> Self {
        self.mcp = McpAttach::Skip;
        self
    }

    /// Attach a pre-connected MCP manager (caller must have merged MCP tools into the registry if needed).
    pub fn mcp(mut self, manager: McpManager) -> Self {
        self.mcp = McpAttach::Inject(manager);
        self
    }

    /// Re-enable auto MCP connect after [`Self::without_mcp`] or [`Self::mcp`].
    pub fn mcp_auto(mut self) -> Self {
        self.mcp = McpAttach::Auto;
        self
    }

    /// Inject a custom [`ContextEngine`] (window size taken from config unless you set it on the engine).
    pub fn context_engine(mut self, context: ContextEngine) -> Self {
        self.context = Some(context);
        self
    }

    /// Inject the durable execution record store.
    pub fn record_writer(mut self, writer: AgentRecordWriter) -> Self {
        self.record_writer = Some(writer);
        self
    }

    /// Inject the session snapshot store used by runtime writeback.
    pub fn session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Inject a runtime model executor instead of wrapping [`ChatClient`].
    pub fn model_executor(mut self, executor: Arc<dyn zene_model_executor::ModelExecutor>) -> Self {
        self.model_executor = Some(executor);
        self
    }

    /// Inject the async approval waiter used when policy returns `Ask`.
    pub fn approval_broker(mut self, broker: zene_permission::SharedApprovalBroker) -> Self {
        self.approval_broker = Some(broker);
        self
    }

    /// Override inference / gateway session id (also reads `ZENE_SESSION_ID` when unset).
    pub fn external_session_id(mut self, id: impl Into<String>) -> Self {
        self.external_session_id = Some(id.into());
        self
    }

    /// Pre-built hook runner; disables config hook loading unless [`Self::load_hooks_from_config`] is set.
    pub fn hooks(mut self, hooks: HookRunner) -> Self {
        self.hooks = Some(hooks);
        self.load_hooks_from_config = false;
        self
    }

    /// Register an in-process host extension hook.
    pub fn extension_hook(mut self, hook: Arc<dyn ExtensionHook>) -> Self {
        self.extensions.push(hook);
        self
    }

    /// Register multiple in-process host extension hooks.
    pub fn extension_hooks(mut self, hooks: Vec<Arc<dyn ExtensionHook>>) -> Self {
        self.extensions.extend(hooks);
        self
    }

    /// Load `.zene/hooks` from config (default when no custom hooks).
    pub fn load_hooks_from_config(mut self, load: bool) -> Self {
        self.load_hooks_from_config = load;
        self
    }

    pub fn permission(mut self, permission: SharedToolPermission) -> Self {
        self.permission = Some(permission);
        self
    }

    pub fn plan_mode(mut self, plan_mode: SharedPlanMode) -> Self {
        self.plan_mode = Some(plan_mode);
        self
    }

    pub fn plan_approval(mut self, prompter: PlanApprovalPrompter) -> Self {
        self.plan_approval = Some(prompter);
        self
    }

    pub fn todos(mut self, todos: SharedTodoStore) -> Self {
        self.todos = Some(todos);
        self
    }

    pub fn ask_user(mut self, prompter: SharedAskUserPrompter) -> Self {
        self.ask_user = Some(prompter);
        self
    }

    pub fn background_tasks(mut self, background: SharedBackgroundTasks) -> Self {
        self.background = Some(background);
        self
    }

    /// Override workspace context inclusion for the system prompt.
    pub fn include_workspace_context(mut self, include: bool) -> Self {
        self.include_workspace_context = Some(include);
        self
    }

    /// Restrict `git push` / `gh` CLI invocations.
    pub fn deny_git_push(mut self, deny: bool) -> Self {
        self.deny_git_push = Some(deny);
        self
    }

    pub async fn build(mut self) -> Result<Agent> {
        let local = self.sandbox;
        let workdir = local.workdir().to_path_buf();
        let include_workspace = self
            .include_workspace_context
            .unwrap_or(self.config.include_workspace_context);
        let workspace_provider = FsWorkspaceProvider::new(workdir.clone());
        let system_prompt = build_system_prompt(
            &self.config.system_prompt,
            &workspace_provider,
            include_workspace,
        );
        self.session.ensure_system_message(&system_prompt);
        let memory_store = FsMemoryStore::new(&workdir);
        let projected = self.session.view().messages;
        if !conversation_has_memory_context(&projected) {
            if let (Some(system), Some(memory)) = (
                projected
                    .first()
                    .filter(|message| message.role == zene_llm::Role::System),
                memory_reminder_from_store(&memory_store),
            ) {
                let existing = system.content.clone().unwrap_or_default();
                self.session
                    .update_system_prefix(&format!("{existing}\n\n{memory}"));
            }
        }

        let client = Arc::new(match self.client {
            Some(client) => client,
            None => ChatClient::from_config(&self.config).await?,
        });
        let record_writer = match self.record_writer {
            Some(writer) => writer,
            None => AgentRecordWriter::for_session(&self.session.meta.id)?,
        };

        let runtime_scope =
            RuntimeScope::agent(self.config.agent_profile, self.config.web_search.clone());
        let mut tools = match self.tools {
            Some(tools) => tools,
            None => runtime_scope.tools(),
        };

        let mcp = match self.mcp {
            McpAttach::Skip => None,
            McpAttach::Inject(manager) => Some(manager),
            McpAttach::Auto => {
                let (mcp, mcp_tools) = McpManager::connect_with_sandbox(&workdir, &local).await?;
                if !mcp_tools.definitions().is_empty() {
                    info!(
                        tool_count = mcp_tools.definitions().len(),
                        "registered MCP tools"
                    );
                    tools.extend(mcp_tools);
                }
                if mcp.is_empty() {
                    None
                } else {
                    Some(mcp)
                }
            }
        };

        let sandbox: Arc<dyn Sandbox> = zene_sandbox::into_arc(local);

        let mut hooks = match self.hooks {
            Some(hooks) => hooks,
            None if self.load_hooks_from_config => {
                let hook_entries = self.config.load_hooks().unwrap_or_else(|err| {
                    warn!(error = %err, "failed to load hooks; continuing without hooks");
                    Vec::new()
                });
                HookRunner::with_bash(hook_specs_from_entries(hook_entries), workdir.clone())
            }
            None => HookRunner::with_bash(Vec::new(), workdir.clone()),
        };
        for ext in self.extensions {
            hooks.add_extension(ext);
        }

        let todos = match self.todos {
            Some(todos) => todos,
            None => shared_todo_store_from(self.session.todos.clone()),
        };

        let mut context = match self.context {
            Some(context) => context,
            None => ContextEngine::new(self.config.compaction.context_window_tokens),
        };

        if let Some(id) = self
            .external_session_id
            .or_else(zene_context::external_session_id_from_env)
        {
            context.set_external_session_id(Some(id));
        }

        let auto_allow_bash = self.config.sandbox.auto_allow_bash && sandbox.is_enforced();
        let permission = match self.permission {
            Some(permission) => permission,
            None => shared_permission_with_rules(
                self.permission_mode,
                permission_rules_from_config(&self.config),
                auto_allow_bash,
                self.deny_git_push,
            ),
        };

        Ok(Agent {
            config: self.config,
            model_executor: self.model_executor.unwrap_or_else(|| {
                Arc::new(zene_model_executor::ChatClientExecutor::new(Arc::clone(
                    &client,
                )))
            }),
            context_model: client,
            runtime_scope,
            tools: Arc::new(tools),
            sandbox,
            session: self.session,
            usage_accumulator: crate::usage::UsageAccumulator::default(),
            context,
            resume_existing_turn: false,
            active_turn: None,
            steer_buffer: Arc::new(Mutex::new(SteerBuffer::default())),
            follow_up_buffer: Arc::new(Mutex::new(FollowUpBuffer::default())),
            system_prompt,
            permission,
            plan_mode: self.plan_mode.unwrap_or_else(shared_plan_mode),
            plan_approval: self
                .plan_approval
                .unwrap_or_else(|| Arc::new(default_plan_approval_prompter)),
            todos,
            ask_user: self.ask_user.unwrap_or_else(default_ask_user_prompter),
            tool_dedup: ToolDedup::new(),
            hooks,
            record_writer,
            session_store: self.session_store.unwrap_or_else(|| {
                if let Ok(url) = std::env::var("CELLZ_URL") {
                    if !url.is_empty() {
                        return Arc::new(zene_session::CellzSessionStore::new(url));
                    }
                }
                Arc::new(FileSessionStore)
            }),
            mcp,
            background: self.background.unwrap_or_else(shared_background_tasks),
            approval_broker: self.approval_broker,
            runtime_approval_waiters: false,
        })
    }
}

pub(crate) fn permission_rules_from_config(config: &ZeneConfig) -> Vec<PermissionRule> {
    config
        .permission_rules
        .to_flat_rules()
        .into_iter()
        .filter_map(|rule| {
            let action = match rule.action.trim().to_lowercase().as_str() {
                "allow" => RuleAction::Allow,
                "deny" => RuleAction::Deny,
                "ask" => RuleAction::Ask,
                _ => return None,
            };
            Some(PermissionRule {
                pattern: rule.pattern,
                action,
            })
        })
        .collect()
}

pub(crate) fn hook_specs_from_entries(entries: Vec<zene_config::HookEntry>) -> Vec<HookSpec> {
    entries
        .into_iter()
        .map(|entry| HookSpec {
            event: entry.event,
            command: entry.command,
        })
        .collect()
}

pub(crate) fn shared_permission_with_rules(
    mode: PermissionMode,
    rules: Vec<PermissionRule>,
    auto_allow_bash: bool,
    deny_git_push: Option<bool>,
) -> SharedToolPermission {
    let mut gate = PermissionGate::new(mode)
        .with_rules(rules)
        .with_auto_allow_bash(auto_allow_bash);
    if let Some(deny) = deny_git_push {
        gate.set_deny_git_push(deny);
    }
    Arc::new(Mutex::new(gate))
}
