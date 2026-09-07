use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::engine::{HookEngine, HookEvent, HookPayload, HookSpec};
use crate::executor::{HookBlock, HookExecutor, HookOutcome};

/// Host-facing in-process extension hook trait.
#[async_trait]
pub trait ExtensionHook: Send + Sync {
    /// Handle a lifecycle event. Return `Ok(HookOutcome::Block(block))` to block/terminate where applicable.
    async fn on_event(&self, payload: &HookPayload) -> Result<HookOutcome> {
        let _ = payload;
        Ok(HookOutcome::Allow)
    }

    /// Optional filter for events this hook cares about. `None` means interested in all events.
    fn interested_events(&self) -> Option<Vec<HookEvent>> {
        None
    }
}

/// Orchestrates hook planning + execution (composition root for hooks).
pub struct HookRunner {
    engine: HookEngine,
    executor: Arc<dyn HookExecutor>,
    extensions: Vec<Arc<dyn ExtensionHook>>,
}

impl HookRunner {
    pub fn new(hooks: Vec<HookSpec>, executor: Arc<dyn HookExecutor>) -> Self {
        Self {
            engine: HookEngine::new(hooks),
            executor,
            extensions: Vec::new(),
        }
    }

    pub fn with_bash(hooks: Vec<HookSpec>, workdir: std::path::PathBuf) -> Self {
        Self::new(
            hooks,
            Arc::new(crate::executor::BashHookExecutor::new(workdir)),
        )
    }

    pub fn with_extension(mut self, extension: Arc<dyn ExtensionHook>) -> Self {
        self.extensions.push(extension);
        self
    }

    pub fn with_extensions(mut self, extensions: Vec<Arc<dyn ExtensionHook>>) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn add_extension(&mut self, extension: Arc<dyn ExtensionHook>) {
        self.extensions.push(extension);
    }

    pub fn extend_specs(&mut self, hooks: Vec<HookSpec>) {
        self.engine.extend(hooks);
    }

    pub fn is_empty(&self) -> bool {
        self.engine.is_empty() && self.extensions.is_empty()
    }

    pub async fn run_pre_tool_use(&self, tool: &str, args: &str) -> Result<Option<HookBlock>> {
        let payload = HookPayload::pre_tool_use(tool, args);

        for ext in &self.extensions {
            if let Some(events) = ext.interested_events() {
                if !events.contains(&HookEvent::PreToolUse) {
                    continue;
                }
            }
            match ext.on_event(&payload).await? {
                HookOutcome::Allow => {}
                HookOutcome::Block(block) => return Ok(Some(block)),
            }
        }

        for request in self.engine.plan_pre_tool_use(tool, args)? {
            match self.executor.run(&request).await? {
                HookOutcome::Allow => {}
                HookOutcome::Block(block) => return Ok(Some(block)),
            }
        }
        Ok(None)
    }

    pub async fn run_post_tool_use(&self, tool: &str, args: &str) {
        let payload = HookPayload::post_tool_use(tool, args);

        for ext in &self.extensions {
            if let Some(events) = ext.interested_events() {
                if !events.contains(&HookEvent::PostToolUse) {
                    continue;
                }
            }
            if let Err(err) = ext.on_event(&payload).await {
                tracing::warn!(tool, error = %err, "PostToolUse extension hook failed");
            }
        }

        if let Ok(planned) = self.engine.plan_post_tool_use(tool, args) {
            for request in planned {
                if let Err(err) = self.executor.run(&request).await {
                    tracing::warn!(tool, error = %err, "PostToolUse command hook failed");
                }
            }
        }
    }

    pub async fn run_before_agent_start(
        &self,
        session_id: &str,
        prompt: &str,
    ) -> Result<Option<HookBlock>> {
        let payload = HookPayload::before_agent_start(session_id, prompt);

        for ext in &self.extensions {
            if let Some(events) = ext.interested_events() {
                if !events.contains(&HookEvent::BeforeAgentStart) {
                    continue;
                }
            }
            match ext.on_event(&payload).await? {
                HookOutcome::Allow => {}
                HookOutcome::Block(block) => return Ok(Some(block)),
            }
        }

        for request in self.engine.plan_before_agent_start(session_id, prompt)? {
            match self.executor.run(&request).await? {
                HookOutcome::Allow => {}
                HookOutcome::Block(block) => return Ok(Some(block)),
            }
        }
        Ok(None)
    }

    pub async fn run_session_before_compact(&self, session_id: &str, reason: &str, tokens: u32) {
        let payload = HookPayload::session_before_compact(session_id, reason, tokens);

        for ext in &self.extensions {
            if let Some(events) = ext.interested_events() {
                if !events.contains(&HookEvent::SessionBeforeCompact) {
                    continue;
                }
            }
            if let Err(err) = ext.on_event(&payload).await {
                tracing::warn!(session_id, reason, error = %err, "SessionBeforeCompact extension hook failed");
            }
        }

        if let Ok(planned) = self
            .engine
            .plan_session_before_compact(session_id, reason, tokens)
        {
            for request in planned {
                if let Err(err) = self.executor.run(&request).await {
                    tracing::warn!(session_id, reason, error = %err, "SessionBeforeCompact command hook failed");
                }
            }
        }
    }

    pub async fn run_context_mutate(&self, session_id: &str, epoch: u64) {
        let payload = HookPayload::context_mutate(session_id, epoch);

        for ext in &self.extensions {
            if let Some(events) = ext.interested_events() {
                if !events.contains(&HookEvent::ContextMutate) {
                    continue;
                }
            }
            if let Err(err) = ext.on_event(&payload).await {
                tracing::warn!(session_id, epoch, error = %err, "ContextMutate extension hook failed");
            }
        }

        if let Ok(planned) = self.engine.plan_context_mutate(session_id, epoch) {
            for request in planned {
                if let Err(err) = self.executor.run(&request).await {
                    tracing::warn!(session_id, epoch, error = %err, "ContextMutate command hook failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::const_new(());

    fn sample_hook(event: &str, script: &str) -> (TempDir, HookRunner) {
        let temp = TempDir::new().expect("tempdir");
        let runner = HookRunner::with_bash(
            vec![HookSpec {
                event: event.into(),
                command: script.into(),
            }],
            temp.path().to_path_buf(),
        );
        (temp, runner)
    }

    struct MockExtension {
        block_on_tool: Option<String>,
        terminate: bool,
    }

    #[async_trait]
    impl ExtensionHook for MockExtension {
        async fn on_event(&self, payload: &HookPayload) -> Result<HookOutcome> {
            if let Some(tool) = &payload.tool {
                if Some(tool) == self.block_on_tool.as_ref() {
                    return Ok(HookOutcome::Block(HookBlock {
                        reason: format!("blocked tool {}", tool),
                        terminate: self.terminate,
                    }));
                }
            }
            Ok(HookOutcome::Allow)
        }
    }

    #[tokio::test]
    async fn pre_tool_use_blocks_on_non_zero_exit() {
        let _guard = TEST_LOCK.lock().await;
        let (_temp, runner) = sample_hook(
            "PreToolUse",
            r#"cat >/dev/null; echo "not allowed" >&2; exit 1"#,
        );
        let block = runner
            .run_pre_tool_use("Write", r#"{"path":"foo.txt"}"#)
            .await
            .expect("run hook")
            .expect("blocked");
        assert_eq!(block.reason, "not allowed");
        assert!(!block.terminate);
    }

    #[tokio::test]
    async fn pre_tool_use_terminates_on_exit_code_2() {
        let _guard = TEST_LOCK.lock().await;
        let (_temp, runner) = sample_hook(
            "PreToolUse",
            r#"cat >/dev/null; echo "security violation" >&2; exit 2"#,
        );
        let block = runner
            .run_pre_tool_use("Bash", r#"{"command":"rm -rf /"}"#)
            .await
            .expect("run hook")
            .expect("blocked");
        assert_eq!(block.reason, "security violation");
        assert!(block.terminate);
    }

    #[tokio::test]
    async fn pre_tool_use_terminates_on_json_output() {
        let _guard = TEST_LOCK.lock().await;
        let (_temp, runner) = sample_hook(
            "PreToolUse",
            r#"cat >/dev/null; echo '{"block":true,"reason":"policy stop","terminate":true}'; exit 0"#,
        );
        let block = runner
            .run_pre_tool_use("Bash", r#"{"command":"something"}"#)
            .await
            .expect("run hook")
            .expect("blocked");
        assert_eq!(block.reason, "policy stop");
        assert!(block.terminate);
    }

    #[tokio::test]
    async fn extension_hook_blocks_and_terminates() {
        let temp = TempDir::new().expect("tempdir");
        let mut runner = HookRunner::with_bash(Vec::new(), temp.path().to_path_buf());
        runner.add_extension(Arc::new(MockExtension {
            block_on_tool: Some("Write".to_string()),
            terminate: true,
        }));

        let block = runner
            .run_pre_tool_use("Write", "{}")
            .await
            .unwrap()
            .expect("blocked");
        assert_eq!(block.reason, "blocked tool Write");
        assert!(block.terminate);

        let allow = runner.run_pre_tool_use("Read", "{}").await.unwrap();
        assert!(allow.is_none());
    }

    #[tokio::test]
    async fn pre_tool_use_allows_success() {
        let _guard = TEST_LOCK.lock().await;
        let (_temp, runner) = sample_hook("PreToolUse", "cat > /dev/null; exit 0");
        let block = runner
            .run_pre_tool_use("Read", r#"{"path":"foo.rs"}"#)
            .await
            .expect("run hook");
        assert!(block.is_none());
    }

    #[tokio::test]
    async fn post_tool_use_does_not_block() {
        let _guard = TEST_LOCK.lock().await;
        let (_temp, runner) =
            sample_hook("PostToolUse", "cat >/dev/null; echo blocked >&2; exit 1");
        runner.run_post_tool_use("Read", "{}").await;
    }

    #[tokio::test]
    async fn hook_receives_json_on_stdin() {
        let _guard = TEST_LOCK.lock().await;
        let temp = TempDir::new().expect("tempdir");
        let script = r#"read payload; echo "$payload" | grep -q '"tool":"Bash"' || exit 2"#;
        let runner = HookRunner::with_bash(
            vec![HookSpec {
                event: "PreToolUse".into(),
                command: script.into(),
            }],
            temp.path().to_path_buf(),
        );
        let block = runner
            .run_pre_tool_use("Bash", r#"{"command":"ls"}"#)
            .await
            .expect("run hook");
        assert!(block.is_none());
    }
}
