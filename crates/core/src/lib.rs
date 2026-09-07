use anyhow::{bail, Context, Result};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;
use zene_config::ZeneConfig;
use zene_context::{ContextDeps, ContextEngine, PrefireClientFactory};
use zene_llm::{ChatClient, Message, TokenUsage, ToolCall};

use zene_mcp::McpManager;
use zene_model_executor::ModelExecutor;
use zene_sandbox::{LocalSandbox, Sandbox};
use zene_session::{
    fork_session, latest_checkpoint_id, load_checkpoint, restore_checkpoint, save_checkpoint,
    AgentRecordWriter, ExecutionCheckpointState, RecordEntry, SessionRecord,
};
pub use zene_session::{RecoveryDisposition, RecoveryExecution, RecoverySnapshot};
pub use zene_tools::AskUserOption;
use zene_tools::{
    shared_todo_store_from, RuntimeScope, SharedAskUserPrompter, SharedBackgroundTasks,
    SharedPlanMode, SharedTodoStore, ToolCatalog, ToolRegistry,
};

mod agent_builder;
mod agent_turn;
mod context_config;
pub use zene_model_executor as model_executor;
mod context_events;
mod context_hooks;
mod events;
mod model_step;
mod plan_mode;
mod prepare_step;
mod subagent;
mod tool_batch;
mod tool_dedup;
mod tool_executor;
pub mod tool_scheduler;
mod turn_session;
mod usage;
mod worktree;

pub use zene_context::{
    compact_session, compact_session_forced, ensure_memory_in_system, estimate_context,
    memory_enabled, memory_root, CompactionResult, ContextWaterLevel, EstimateMode,
    EstimateProvider, InputLadderStage, TiktokenEncoding, TokenEstimator,
};

use crate::agent_turn::AgentTurnPorts;
pub use agent_builder::AgentBuilder;
pub use events::{emit_event, runtime_event_handler, AgentEvent, EventHandler};
use plan_mode::tool_visible_in_definitions;
pub use plan_mode::PlanApprovalPrompter;
pub use subagent::{run_subagent, CoreSubagentRunner};
pub use tool_dedup::{append_reminder, ToolDedup};
pub use tool_scheduler::{classify_tool_accesses, ToolScheduler};
pub use zene_hooks::{
    ExtensionHook, HookBlock, HookEvent, HookOutcome, HookPayload, HookRunner, HookSpec,
};
pub use zene_permission::{
    approve_tool_call, deny_cloud_github_cli, deny_git_cli, policy_denied,
    policy_denied_with_git_push, resolve_permission, ApprovalBroker, ApprovalRequest,
    AutoApprovalBroker, PermissionGate, PermissionMode, PermissionPrompter, PermissionRule,
    PolicyDecision, PromptChoice, RuleAction, SharedApprovalBroker, SharedToolPermission,
    TerminalApprovalBroker, ToolPermission,
};
use zene_turn::{
    FollowUpBuffer, QueueMode, RuntimeEventHandler, SessionId, SteerBuffer, StepId, TurnId,
    TurnState,
};
pub use zene_workspace::{build_system_prompt, FsWorkspaceProvider, WorkspaceProvider};

pub use worktree::ensure_session_worktree;

pub struct Agent {
    config: ZeneConfig,
    context_model: Arc<dyn zene_context::ContextModel>,
    model_executor: Arc<dyn ModelExecutor>,
    /// Root capability scope (depth 0). Tools may be overridden after scope build.
    runtime_scope: RuntimeScope,
    tools: Arc<ToolRegistry>,
    sandbox: Arc<dyn Sandbox>,
    session: SessionRecord,
    usage_accumulator: usage::UsageAccumulator,
    context: ContextEngine,
    /// Set when resuming a safe model-boundary turn; cleared on prepare_turn.
    resume_existing_turn: bool,
    active_turn: Option<TurnState>,
    steer_buffer: Arc<Mutex<SteerBuffer>>,
    follow_up_buffer: Arc<Mutex<FollowUpBuffer>>,
    system_prompt: String,
    permission: SharedToolPermission,
    plan_mode: SharedPlanMode,
    plan_approval: PlanApprovalPrompter,
    todos: SharedTodoStore,
    ask_user: SharedAskUserPrompter,
    tool_dedup: ToolDedup,
    hooks: HookRunner,
    record_writer: AgentRecordWriter,
    session_store: Arc<dyn zene_session::SessionStore>,
    mcp: Option<McpManager>,
    background: SharedBackgroundTasks,
    approval_broker: Option<zene_permission::SharedApprovalBroker>,
    runtime_approval_waiters: bool,
}

pub struct PromptOptions {
    pub stream: bool,
    pub cancel: Option<CancellationToken>,
    pub event_handler: Option<EventHandler>,
    /// Shared runtime event sink; legacy event_handler remains supported.
    pub runtime_event_handler: Option<RuntimeEventHandler>,
    /// When true, suppress stdout/stderr tool and stream printing (for TUI).
    pub quiet: bool,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            stream: true,
            cancel: None,
            event_handler: None,
            runtime_event_handler: None,
            quiet: false,
        }
    }
}

impl Agent {
    pub async fn new(
        config: ZeneConfig,
        sandbox: LocalSandbox,
        session: SessionRecord,
        permission_mode: PermissionMode,
    ) -> Result<Self> {
        AgentBuilder::new(config, sandbox, session, permission_mode)
            .build()
            .await
    }

    /// Override inference session id (e.g. Cloud run_id for gateway linkage).
    pub fn set_external_session_id(&mut self, id: Option<String>) {
        self.context.set_external_session_id(id);
    }

    /// Access the agent's hook runner.
    pub fn hooks(&self) -> &HookRunner {
        &self.hooks
    }

    /// Access the agent's hook runner mutably.
    pub fn hooks_mut(&mut self) -> &mut HookRunner {
        &mut self.hooks
    }

    /// Register an in-process host extension hook.
    pub fn add_extension_hook(&mut self, hook: Arc<dyn ExtensionHook>) {
        self.hooks.add_extension(hook);
    }

    pub fn is_plan_mode_active(&self) -> bool {
        self.plan_mode.lock().is_active()
    }

    pub fn enter_plan_mode(&mut self) {
        self.enter_plan_mode_with_handler(&None);
    }

    pub fn leave_plan_mode(&mut self) {
        self.leave_plan_mode_with_handler(&None);
    }

    /// Set ACP/session mode. Returns the active mode id (`default` or `plan`).
    pub fn set_session_mode(&mut self, mode_id: &str) -> Result<String> {
        match mode_id {
            "plan" | "architect" => {
                self.enter_plan_mode();
                Ok("plan".into())
            }
            "default" | "agent" | "code" | "ask" => {
                self.leave_plan_mode();
                Ok("default".into())
            }
            other => bail!("unknown session mode: {other}"),
        }
    }

    pub fn current_session_mode(&self) -> &'static str {
        if self.is_plan_mode_active() {
            "plan"
        } else {
            "default"
        }
    }

    fn enter_plan_mode_with_handler(&mut self, handler: &Option<EventHandler>) {
        let should_sync = {
            let mut state = self.plan_mode.lock();
            if state.is_active() {
                false
            } else {
                state.enter();
                true
            }
        };
        if should_sync {
            self.session.record_mode_changed("plan");
            emit_event(
                handler,
                AgentEvent::ModeChanged {
                    mode_id: "plan".into(),
                },
            );
        }
    }

    fn leave_plan_mode_with_handler(&mut self, handler: &Option<EventHandler>) {
        let should_sync = {
            let mut state = self.plan_mode.lock();
            if !state.is_active() {
                false
            } else {
                state.exit();
                true
            }
        };
        if should_sync {
            self.session.record_mode_changed("default");
            emit_event(
                handler,
                AgentEvent::ModeChanged {
                    mode_id: "default".into(),
                },
            );
        }
    }

    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        let active = self.is_plan_mode_active();
        if !active {
            return true;
        }
        self.plan_mode.lock().is_tool_allowed(tool_name)
    }

    /// Activate registered tools for subsequent model requests.
    pub fn activate_tools(&self, names: &[String]) -> Vec<String> {
        self.tools.activate_many(names.iter())
    }

    /// Deactivate tools from subsequent model requests without unregistering them.
    pub fn deactivate_tools(&self, names: &[String]) -> Vec<String> {
        self.tools.deactivate_many(names.iter())
    }

    pub fn active_tool_names(&self) -> Vec<String> {
        self.tools.active_tool_names()
    }

    fn tool_definitions_for_llm(&self) -> Vec<zene_llm::ToolDefinition> {
        let active = self.is_plan_mode_active();
        ToolCatalog::definitions(self.tools.as_ref())
            .into_iter()
            .filter(|def| tool_visible_in_definitions(&def.name, active))
            .collect()
    }

    pub fn execution_record_writer(&self) -> AgentRecordWriter {
        self.record_writer.clone()
    }

    /// Mark the next prompt as a safe model-boundary resume (no new user message).
    pub fn set_resume_existing_turn(&mut self, resume: bool) {
        self.resume_existing_turn = resume;
    }

    pub fn runtime_approval_waiters(&self) -> bool {
        self.runtime_approval_waiters
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(mcp) = self.mcp.as_mut() {
            mcp.disconnect().await?;
        }
        let session_id = self.context.metadata(&self.session).session_id;
        zene_context::close_session(&session_id).await;
        self.sandbox.shutdown().await?;
        Ok(())
    }

    pub fn config(&self) -> &ZeneConfig {
        &self.config
    }

    pub async fn switch_model(
        &mut self,
        model: &str,
        provider: Option<String>,
        base_url: Option<String>,
        api_key: Option<String>,
    ) -> Result<()> {
        if let Some(p) = provider {
            self.config.provider = p;
        }
        self.config.model = model.to_string();
        if let Some(url) = base_url {
            if self.config.provider.trim().to_lowercase() == "anthropic" {
                self.config.anthropic_base_url = Some(url);
            } else {
                self.config.base_url = url;
            }
        }
        if let Some(key) = api_key {
            if self.config.provider.trim().to_lowercase() == "anthropic" {
                self.config.anthropic_api_key = Some(key);
            } else {
                self.config.api_key = Some(key);
            }
        }

        self.session.record_model_changed(model);
        self.config.refresh_model_context_window();
        self.context
            .set_window(self.config.compaction.context_window_tokens);

        // Recreate the client and context model.
        let client = Arc::new(zene_llm::ChatClient::from_config(&self.config).await?);
        self.context_model = client.clone();
        self.model_executor = Arc::new(zene_model_executor::ChatClientExecutor::new(client));
        self.config
            .persist_connection_settings()
            .context("save model settings to ~/.zene/config.toml")?;
        Ok(())
    }

    pub fn context_water(&self) -> &ContextWaterLevel {
        self.context.water()
    }

    /// Manually compact the conversation (`/compact [hint]`).
    pub async fn compact_now(
        &mut self,
        user_hint: Option<&str>,
    ) -> Result<Option<CompactionResult>> {
        self.sync_todos_to_session();
        let tools = self.tool_definitions_for_llm();
        let estimator = self.token_estimator();
        let background_tasks = self.background.lock().list();
        let hooks = context_hooks::ZeneContextHooks::new(
            &self.session,
            &background_tasks,
            self.is_plan_mode_active(),
        );
        let compaction_config = context_config::context_compaction_config(&self.config.compaction);
        let mut handler = context_events::AgentContextHandler::new(
            self.context_model.as_ref(),
            &self.config.model,
            self.sandbox.workdir(),
        );
        let prefire_factory = self.prefire_client_factory();
        let mut deps = ContextDeps {
            session: &mut self.session,
            compaction_config: &compaction_config,
            model: &self.config.model,
            client: self.context_model.as_ref(),
            hooks: Some(&hooks),
            system_prompt: &self.system_prompt,
            estimator: &estimator,
            handler: &mut handler,
            prefire_client_factory: prefire_factory,
        };
        let result = self
            .context
            .compact_forced(&mut deps, &tools, user_hint)
            .await;
        match &result {
            Ok(forced) => {
                if let Some(compact_result) = &forced.compaction {
                    self.record_compaction(compact_result)?;
                }
                self.save_session()?;
                Ok(forced.compaction.clone())
            }
            Err(_) => result.map(|forced| forced.compaction),
        }
    }

    /// Human-readable context report for `/context` (grok-aligned).
    pub fn context_report(&self) -> String {
        let water = self.context.water();
        let cfg = context_config::context_compaction_config(&self.config.compaction);
        let threshold_pct = ContextWaterLevel::auto_compact_threshold_percent(&cfg);
        let threshold_tokens = ContextWaterLevel::threshold_tokens(&cfg);
        let used = water.effective_tokens();
        let window = water.context_window_tokens.max(1);
        let mut lines = vec![
            format!(
                "context: {}% ({} / {} tokens)",
                water.usage_percent(),
                used,
                window
            ),
            format!(
                "auto-compact at {}% ({} tokens)",
                threshold_pct, threshold_tokens
            ),
            format!(
                "sources: usage={} estimate={}",
                water
                    .last_prompt_tokens
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                water
                    .last_estimate_tokens
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into())
            ),
            format!("messages: {}", self.session.view().messages.len()),
            format!("model: {}", self.config.model),
            format!("context_epoch: {}", self.context.epoch()),
        ];
        let delivery = zene_context::delivery_mode_from_env();
        lines.push(format!(
            "delivery: {} (gateway_prefix_len={})",
            delivery.as_str(),
            self.context.gateway_prefix_len()
        ));
        let estimator = self.token_estimator();
        let tools = self.tool_definitions_for_llm();
        let explain = self
            .context
            .try_explain_projection(&self.session, &tools, &estimator)
            .unwrap_or_else(|err| {
                let mut explain =
                    self.context
                        .explain_projection(&self.session, &tools, &estimator);
                explain.fallback_reason = Some(format!("strict_projection_error: {err}"));
                explain.used_materialized_fallback = true;
                explain
            });
        lines.extend([
            format!(
                "projection: source_messages={} projected_messages={} source_events={} active_events={}",
                explain.source_message_count,
                explain.projected_message_count,
                explain.source_event_count,
                explain.active_event_count,
            ),
            format!(
                "projection_path: branch={} start_sequence={}",
                explain.active_branch_id.as_deref().unwrap_or("-"),
                explain
                    .active_path_start_sequence
                    .map(|sequence| sequence.to_string())
                    .unwrap_or_else(|| "-".into()),
            ),
            format!(
                "projection_fallback: used={} reason={} cache_drift={}",
                explain.used_materialized_fallback,
                explain.fallback_reason.as_deref().unwrap_or("-"),
                explain.cache_drift_detected,
            ),
            format!(
                "projection_decorations: injected={} delivery={} tail_start={} estimate_tokens={}",
                if explain.injected.is_empty() {
                    "-".to_string()
                } else {
                    explain.injected.join(",")
                },
                explain.delivery.as_str(),
                explain
                    .delivery_tail_start
                    .map(|start| start.to_string())
                    .unwrap_or_else(|| "-".into()),
                explain.estimate_tokens,
            ),
            format!(
                "prefix_cache: break={} prefix_end={} tail={} cached={} gateway_hit={} reprocessed_est={}",
                explain.prefix_cache.break_kind,
                explain.prefix_cache.prefix_end,
                explain.prefix_cache.tail_decoration_count,
                explain
                    .prefix_cache
                    .cached_tokens
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "-".into()),
                explain
                    .prefix_cache
                    .gateway_hit_tokens
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "-".into()),
                explain
                    .prefix_cache
                    .unchanged_reprocessed_est
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "-".into()),
            ),
        ]);
        if water.auto_compact_suppressed {
            lines.push(
                "auto-compact: suppressed (last summarize failed; use /compact to retry)"
                    .to_string(),
            );
        }
        if self.context.prefire_has_cache() {
            lines.push("prefire: NOTE₁ cached (two-pass ready)".to_string());
        } else if self.context.prefire_in_flight() {
            lines.push("prefire: pass1 in flight".to_string());
        }
        if memory_enabled() {
            let root = memory_root(self.sandbox.workdir());
            lines.push(format!("memory: {} (ZENE_MEMORY)", root.display()));
        } else {
            lines.push("memory: disabled".to_string());
        }
        lines.join("\n")
    }

    /// Rewind to the latest compaction checkpoint (or a specific id).
    pub fn rewind_to_checkpoint(&mut self, checkpoint_id: Option<&str>) -> Result<String> {
        let id = match checkpoint_id {
            Some(id) => id.to_string(),
            None => latest_checkpoint_id(&self.session.meta.id)?
                .ok_or_else(|| anyhow::anyhow!("no compaction checkpoint available to rewind"))?,
        };
        let checkpoint = load_checkpoint(&self.session.meta.id, &id)?;
        restore_checkpoint(&mut self.session, &checkpoint);
        self.record_writer.append(&RecordEntry::Rewound {
            checkpoint_id: checkpoint.id.clone(),
            target_sequence: Some(checkpoint.event_sequence),
            ts: chrono::Utc::now(),
        })?;
        self.session.ensure_system_message(&self.system_prompt);
        if let Some(tokens) = self.session.context_tokens_used {
            self.context.restore_water_from_session(tokens);
        }
        self.context.clear_prefire();
        self.save_session()?;
        Ok(id)
    }

    /// Fork the current session into a new saved session and switch to it.
    pub fn fork_session(&mut self) -> Result<String> {
        let workdir = self.sandbox.workdir().to_path_buf();
        let forked = fork_session(&self.session, &workdir);
        let id = forked.meta.id.clone();
        self.session_store.save(&forked)?;
        let _ = save_checkpoint(&forked, "fork");
        self.session = forked;
        self.record_writer = AgentRecordWriter::for_session(&id)?;
        self.todos = shared_todo_store_from(self.session.todos.clone());
        self.context.clear_prefire();
        Ok(id)
    }

    pub(crate) fn save_session(&self) -> Result<()> {
        self.session_store.save(&self.session)
    }

    pub fn session(&self) -> &SessionRecord {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut SessionRecord {
        &mut self.session
    }

    pub fn turn_usage(&self) -> &TokenUsage {
        self.usage_accumulator.total()
    }

    pub fn is_turn_active(&self) -> bool {
        self.active_turn.is_some()
    }

    pub fn pending_steer_count(&self) -> usize {
        self.steer_buffer.lock().len()
    }

    pub fn pending_follow_up_count(&self) -> usize {
        self.follow_up_buffer.lock().len()
    }

    pub fn steer_buffer(&self) -> Arc<Mutex<SteerBuffer>> {
        Arc::clone(&self.steer_buffer)
    }

    /// Queue a steering message for the next model call.
    pub fn queue_steer(&self, text: &str) -> Result<()> {
        if !self.is_turn_active() {
            return Err(zene_turn::steer_requires_active_turn());
        }
        let text = text.trim();
        if text.is_empty() {
            anyhow::bail!("steer message cannot be empty");
        }
        self.steer_buffer.lock().push(text.to_string());
        Ok(())
    }

    pub fn steer(&mut self, text: &str) -> Result<()> {
        self.queue_steer(text)
    }

    pub fn follow_up(&self, text: &str) -> Result<()> {
        if !self.is_turn_active() {
            return Err(zene_turn::steer_requires_active_turn());
        }
        let text = text.trim();
        if text.is_empty() {
            anyhow::bail!("follow-up message cannot be empty");
        }
        self.follow_up_buffer.lock().push(text.to_string());
        Ok(())
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steer_buffer.lock().set_mode(mode);
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_buffer.lock().set_mode(mode);
    }

    pub fn clear_queues(&self) {
        self.steer_buffer.lock().clear();
        self.follow_up_buffer.lock().clear();
    }

    fn token_estimator(&self) -> TokenEstimator {
        TokenEstimator::for_provider(
            EstimateProvider::from_name(&self.config.provider),
            &self.config.model,
            self.config.chars_per_token_for_model(),
        )
    }

    fn prefire_client_factory(&self) -> Option<PrefireClientFactory> {
        let config = self.config.clone();
        Some(std::sync::Arc::new(move || {
            let config = config.clone();
            Box::pin(async move {
                let client = ChatClient::from_config(&config).await?;
                Ok(std::sync::Arc::new(client) as std::sync::Arc<dyn zene_context::ContextModel>)
            })
        }))
    }

    /// Replace the permission gate (e.g. TUI custom prompter).
    ///
    /// Re-applies sandbox `auto_allow_bash` and config permission rules so a TUI
    /// prompter swap does not drop sandbox-linked policy.
    pub fn set_permission_gate(&mut self, mut gate: PermissionGate) {
        gate.set_auto_allow_bash(self.config.sandbox.auto_allow_bash && self.sandbox.is_enforced());
        gate.set_rules(agent_builder::permission_rules_from_config(&self.config));
        self.permission = Arc::new(Mutex::new(gate));
    }

    /// Inject the async approval waiter used when policy returns `Ask`.
    pub fn set_approval_broker(&mut self, broker: zene_permission::SharedApprovalBroker) {
        self.approval_broker = Some(broker);
    }

    /// Ask the runtime actor to own approval waiters for this session.
    ///
    /// Transports then send `RuntimeCommand::Approval` (via `zene-runtime` /
    /// `zene-agent-runtime`) instead of injecting an ACP/Cloud-specific broker.
    pub fn enable_runtime_approval_waiters(&mut self) {
        self.runtime_approval_waiters = true;
    }

    /// Replace the AskUserQuestion prompter (e.g. TUI modal).
    pub fn set_ask_user_prompter(&mut self, prompter: SharedAskUserPrompter) {
        self.ask_user = prompter;
    }

    pub async fn prompt(&mut self, user_input: &str, options: PromptOptions) -> Result<String> {
        zene_turn::begin_turn(&mut self.active_turn)?;
        let cancel = options.cancel.clone();
        let turn_id = self
            .active_turn
            .as_ref()
            .map(|t| t.turn_id)
            .expect("turn just started");

        self.record_writer.append(&RecordEntry::TurnPrompt {
            turn_id: turn_id.to_string(),
            prompt: user_input.to_string(),
            ts: chrono::Utc::now(),
        })?;
        let turn_event = self
            .session
            .record_turn_started(&turn_id.to_string(), user_input);
        self.record_writer.append_execution_link(
            &format!("{turn_id}/turn/started"),
            &turn_event.id,
            turn_event.sequence,
        )?;

        let event_handler = merge_event_handler(
            &self.record_writer,
            self.session.meta.id.clone(),
            options.event_handler.clone(),
            options.runtime_event_handler.clone(),
        );

        info!(turn_id = %turn_id, "turn_start");
        emit_event(&event_handler, AgentEvent::TurnStart { turn_id });

        if let Some(block) = self
            .hooks
            .run_before_agent_start(&self.session.meta.id, user_input)
            .await?
        {
            emit_event(
                &event_handler,
                AgentEvent::Error {
                    message: format!("Agent start blocked: {}", block.reason),
                },
            );
            emit_event(&event_handler, AgentEvent::TurnEnd { turn_id, steps: 0 });
            self.session
                .record_turn_ended(&turn_id.to_string(), 0, "blocked");
            bail!("agent start blocked by hook: {}", block.reason);
        }

        emit_event(
            &event_handler,
            AgentEvent::LifecycleEvent {
                event: "before_agent_start".to_string(),
                payload: serde_json::json!({
                    "sessionId": self.session.meta.id,
                    "prompt": user_input,
                })
                .to_string(),
            },
        );

        let run_options = PromptOptions {
            stream: options.stream,
            cancel: options.cancel,
            event_handler,
            runtime_event_handler: None,
            quiet: options.quiet,
        };
        let result = self
            .run_turn(user_input, &run_options, cancel.as_ref())
            .await;

        let steps = self.active_turn.as_ref().map(|t| t.step).unwrap_or(0);
        let was_cancelled = cancel.is_some_and(|token| token.is_cancelled())
            || result
                .as_ref()
                .err()
                .is_some_and(|err| err.to_string().contains("aborted"));
        match &result {
            Ok(_) => {
                info!(turn_id = %turn_id, steps, "turn_end");
                emit_event(
                    &run_options.event_handler,
                    AgentEvent::TurnEnd { turn_id, steps },
                );
                self.session
                    .record_turn_ended(&turn_id.to_string(), steps, "completed");
            }
            Err(err) => {
                if err.to_string().contains("aborted") {
                    info!(turn_id = %turn_id, steps, "turn_end");
                } else {
                    info!(turn_id = %turn_id, steps, error = %err, "turn_end");
                    emit_event(
                        &run_options.event_handler,
                        AgentEvent::Error {
                            message: err.to_string(),
                        },
                    );
                }
                emit_event(
                    &run_options.event_handler,
                    AgentEvent::TurnEnd { turn_id, steps },
                );
                self.session.record_turn_ended(
                    &turn_id.to_string(),
                    steps,
                    if was_cancelled { "cancelled" } else { "failed" },
                );
            }
        }

        let terminal_state = if was_cancelled {
            ExecutionCheckpointState::TurnCancelled
        } else if result.is_ok() {
            ExecutionCheckpointState::TurnCompleted
        } else {
            ExecutionCheckpointState::Failed
        };
        self.append_terminal_checkpoint(turn_id, terminal_state)?;
        zene_turn::end_turn(&mut self.active_turn);
        result
    }

    async fn run_turn(
        &mut self,
        user_input: &str,
        options: &PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<String> {
        let mut ports = AgentTurnPorts::new(self);
        zene_turn::TurnEngine::new(&mut ports)
            .run(zene_turn::TurnRequest::new(user_input, options, cancel))
            .await
            .map(|outcome| outcome.final_text)
    }

    fn sync_todos_to_session(&mut self) {
        let store = self.todos.lock();
        self.session.todos = store.to_items();
    }

    fn inject_pending_steer(&mut self, options: &PromptOptions) -> Result<bool> {
        Ok(turn_session::inject_pending_steer(
            self.runtime_scope.session_policy.steer,
            &self.steer_buffer,
            &mut self.session,
            options,
        ))
    }

    fn inject_pending_follow_up(&mut self, options: &PromptOptions) -> Result<bool> {
        let messages = self.follow_up_buffer.lock().take_all();
        if messages.is_empty() {
            return Ok(false);
        }
        for text in messages {
            emit_event(
                &options.event_handler,
                AgentEvent::SteerInput { text: text.clone() },
            );
            self.session.push_message(Message::user(text));
        }
        Ok(true)
    }

    pub(crate) fn record_step_started(&mut self) -> Result<()> {
        turn_session::record_step_started(
            &mut self.session,
            &self.record_writer,
            self.active_turn.as_ref(),
        )
    }

    pub(crate) async fn prepare_step_context(
        &mut self,
        options: &PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<zene_turn::PreparedContext> {
        let plan_mode_active = self.is_plan_mode_active();
        prepare_step::prepare_step_context(
            prepare_step::PrepareStepDeps {
                config: &self.config,
                context_model: self.context_model.as_ref(),
                context: &mut self.context,
                session: &mut self.session,
                system_prompt: &self.system_prompt,
                workdir: self.sandbox.workdir(),
                tools: self.tools.as_ref(),
                background: &self.background,
                todos: &self.todos,
                tool_policy: self.runtime_scope.tool_policy,
                plan_mode_active,
                record_writer: &self.record_writer,
            },
            options,
            cancel,
        )
        .await
    }

    pub(crate) async fn invoke_model(
        &mut self,
        context: zene_turn::PreparedContext,
        options: &PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<zene_turn::StepResult> {
        let plan_mode_active = self.is_plan_mode_active();
        model_step::run_model_step(
            model_step::ModelStepDeps {
                config: &self.config,
                model_executor: self.model_executor.as_ref(),
                context_model: self.context_model.as_ref(),
                context: &mut self.context,
                session: &mut self.session,
                system_prompt: &self.system_prompt,
                workdir: self.sandbox.workdir(),
                background: &self.background,
                todos: &self.todos,
                plan_mode_active,
                record_writer: &self.record_writer,
            },
            context,
            options,
            cancel,
        )
        .await
    }

    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        options: &PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<zene_turn::ToolBatchOutcome> {
        let (turn_id, step_id) = self
            .active_turn
            .as_ref()
            .map(|turn| {
                (
                    Some(turn.turn_id.to_string()),
                    turn.step_id.map(|step_id| step_id.to_string()),
                )
            })
            .unwrap_or((None, None));
        tool_batch::run_tool_batch(
            tool_batch::ToolBatchDeps {
                config: &self.config,
                tools: Arc::clone(&self.tools),
                sandbox: Arc::clone(&self.sandbox),
                permission: Arc::clone(&self.permission),
                approval_broker: self.approval_broker.clone(),
                plan_mode: Arc::clone(&self.plan_mode),
                plan_approval: &self.plan_approval,
                todos: Arc::clone(&self.todos),
                ask_user: Arc::clone(&self.ask_user),
                background: Arc::clone(&self.background),
                runtime_scope: &self.runtime_scope,
                hooks: &self.hooks,
                session: &mut self.session,
                record_writer: &self.record_writer,
                tool_dedup: &mut self.tool_dedup,
                turn_id,
                step_id,
            },
            tool_calls,
            options,
            cancel,
        )
        .await
    }

    fn record_compaction(&self, result: &CompactionResult) -> Result<()> {
        self.record_writer.append(&RecordEntry::Compaction {
            reason: result.reason.clone(),
            compacted_count: result.compacted_count,
            tokens_before: Some(result.stats.tokens_before),
            tokens_after: Some(result.stats.tokens_after),
            ts: chrono::Utc::now(),
        })
    }
}

fn merge_event_handler(
    record_writer: &AgentRecordWriter,
    session_id: String,
    user_handler: Option<EventHandler>,
    runtime_handler: Option<RuntimeEventHandler>,
) -> Option<EventHandler> {
    let record_writer = record_writer.clone();
    let shared = runtime_event_handler(
        SessionId::from_string(session_id),
        user_handler,
        runtime_handler,
    );
    let scope = Arc::new(Mutex::new((None, None)));
    Some(Arc::new(move |event| {
        {
            let mut current = scope.lock();
            match &event {
                AgentEvent::TurnStart { turn_id } => {
                    current.0 = Some(*turn_id);
                    current.1 = None;
                }
                AgentEvent::StepBegin {
                    turn_id, step_id, ..
                } => {
                    current.0 = Some(*turn_id);
                    current.1 = Some(*step_id);
                }
                AgentEvent::TurnEnd { .. } => current.1 = None,
                _ => {}
            }
        }
        if let Some(entry) = record_entry_from_agent_event(&event) {
            let _ = record_writer.append(&entry);
        }
        let (turn_id, step_id) = {
            let current = scope.lock();
            (current.0, current.1)
        };
        // Tool boundaries are persisted synchronously by `run_tools`, where
        // failures can prevent side effects or be returned after execution.
        // The event handler is intentionally best effort for UI/runtime
        // publication and the remaining non-tool checkpoints.
        if !matches!(
            &event,
            AgentEvent::ToolCall { .. } | AgentEvent::ToolResult { .. }
        ) {
            for checkpoint in execution_checkpoints_from_agent_event(&event, turn_id, step_id) {
                let _ = record_writer.append_execution_checkpoint(&checkpoint);
            }
        }
        shared(event);
    }))
}

fn execution_checkpoints_from_agent_event(
    event: &AgentEvent,
    current_turn: Option<TurnId>,
    current_step: Option<StepId>,
) -> Vec<RecordEntry> {
    let ts = chrono::Utc::now();
    let turn_key = current_turn
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unknown".into());
    let step_key = current_step
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unknown".into());
    let checkpoint = match event {
        AgentEvent::TurnStart { turn_id } => Some(RecordEntry::ExecutionCheckpoint {
            turn_id: turn_id.to_string(),
            step_id: None,
            tool_call_id: None,
            state: ExecutionCheckpointState::TurnStarted,
            idempotency_key: format!("{turn_id}/turn/started"),
            context_epoch: None,
            model_request_hash: None,
            ts,
        }),
        AgentEvent::StepBegin {
            turn_id, step_id, ..
        } => Some(RecordEntry::ExecutionCheckpoint {
            turn_id: turn_id.to_string(),
            step_id: Some(step_id.to_string()),
            tool_call_id: None,
            state: ExecutionCheckpointState::StepStarted,
            idempotency_key: format!("{turn_id}/{step_id}/started"),
            context_epoch: None,
            model_request_hash: None,
            ts,
        }),
        AgentEvent::ToolCall { id, .. } => Some(RecordEntry::ExecutionCheckpoint {
            turn_id: turn_key.clone(),
            step_id: current_step.map(|id| id.to_string()),
            tool_call_id: Some(id.clone()),
            state: ExecutionCheckpointState::ToolStarted,
            idempotency_key: format!("tool/{turn_key}/{step_key}/{id}/started"),
            context_epoch: None,
            model_request_hash: None,
            ts,
        }),
        AgentEvent::ToolResult { id, .. } => Some(RecordEntry::ExecutionCheckpoint {
            turn_id: turn_key.clone(),
            step_id: current_step.map(|id| id.to_string()),
            tool_call_id: Some(id.clone()),
            state: ExecutionCheckpointState::ToolCompleted,
            idempotency_key: format!("tool/{turn_key}/{step_key}/{id}/completed"),
            context_epoch: None,
            model_request_hash: None,
            ts,
        }),
        AgentEvent::TurnEnd { .. } => None,
        AgentEvent::Error { message } => Some(RecordEntry::ExecutionCheckpoint {
            turn_id: turn_key.clone(),
            step_id: current_step.map(|id| id.to_string()),
            tool_call_id: None,
            state: ExecutionCheckpointState::Failed,
            idempotency_key: format!("error/{turn_key}/{step_key}/{message}"),
            context_epoch: None,
            model_request_hash: None,
            ts,
        }),
        AgentEvent::TextDelta { .. }
        | AgentEvent::ThoughtDelta { .. }
        | AgentEvent::ModeChanged { .. }
        | AgentEvent::UsageUpdate { .. }
        | AgentEvent::ProjectionReady { .. }
        | AgentEvent::SteerInput { .. }
        | AgentEvent::LifecycleEvent { .. } => None,
    };
    checkpoint.into_iter().collect()
}

impl Agent {
    fn append_terminal_checkpoint(
        &mut self,
        turn_id: TurnId,
        state: ExecutionCheckpointState,
    ) -> Result<()> {
        let state_name = match &state {
            ExecutionCheckpointState::TurnCancelled => "cancelled",
            ExecutionCheckpointState::TurnCompleted => "completed",
            ExecutionCheckpointState::Failed => "failed",
            _ => "terminal",
        };
        let idempotency_key = format!("{turn_id}/turn/{state_name}");
        let event = self.session.record_checkpoint(
            Some(&turn_id.to_string()),
            None,
            None,
            state_name,
            &idempotency_key,
        );
        self.record_writer
            .append_execution_link(&idempotency_key, &event.id, event.sequence)?;
        self.record_writer
            .append_execution_checkpoint(&RecordEntry::ExecutionCheckpoint {
                turn_id: turn_id.to_string(),
                step_id: None,
                tool_call_id: None,
                state,
                idempotency_key,
                context_epoch: Some(self.context.epoch()),
                model_request_hash: None,
                ts: chrono::Utc::now(),
            })?;
        Ok(())
    }
}

fn record_entry_from_agent_event(event: &AgentEvent) -> Option<RecordEntry> {
    let ts = chrono::Utc::now();
    match event {
        AgentEvent::StepBegin {
            turn_id,
            step_id,
            step,
        } => Some(RecordEntry::StepBegin {
            turn_id: turn_id.to_string(),
            step_id: step_id.to_string(),
            step: *step,
            ts,
        }),
        AgentEvent::ToolCall {
            name, arguments, ..
        } => Some(RecordEntry::ToolCall {
            name: name.clone(),
            arguments: arguments.clone(),
            ts,
        }),
        AgentEvent::ToolResult {
            name,
            content,
            is_error,
            ..
        } => Some(RecordEntry::ToolResult {
            name: name.clone(),
            content: content.clone(),
            is_error: *is_error,
            ts,
        }),
        AgentEvent::TurnEnd { turn_id, steps } => Some(RecordEntry::TurnEnd {
            turn_id: turn_id.to_string(),
            steps: *steps,
            ts,
        }),
        AgentEvent::Error { message } => Some(RecordEntry::Error {
            message: message.clone(),
            ts,
        }),
        AgentEvent::TurnStart { .. }
        | AgentEvent::TextDelta { .. }
        | AgentEvent::ThoughtDelta { .. }
        | AgentEvent::SteerInput { .. }
        | AgentEvent::ModeChanged { .. }
        | AgentEvent::UsageUpdate { .. }
        | AgentEvent::ProjectionReady { .. }
        | AgentEvent::LifecycleEvent { .. } => None,
    }
}

#[cfg(test)]
mod execution_checkpoint_tests {
    use super::*;
    use zene_session::ExecutionCheckpointState;

    #[test]
    fn checkpoint_projection_preserves_turn_step_and_tool_scope() {
        let turn_id = TurnId::new();
        let step_id = StepId::new();
        let turn_text = turn_id.to_string();
        let step_text = step_id.to_string();
        let tool_id = "call-1";
        let checkpoints = execution_checkpoints_from_agent_event(
            &AgentEvent::ToolCall {
                id: tool_id.into(),
                name: "Read".into(),
                arguments: "{}".into(),
            },
            Some(turn_id),
            Some(step_id),
        );
        assert_eq!(checkpoints.len(), 1);
        let RecordEntry::ExecutionCheckpoint {
            turn_id: recorded_turn,
            step_id: recorded_step,
            tool_call_id,
            state,
            idempotency_key,
            ..
        } = &checkpoints[0]
        else {
            panic!("expected execution checkpoint");
        };
        assert_eq!(recorded_turn, &turn_text);
        assert_eq!(recorded_step.as_deref(), Some(step_text.as_str()));
        assert_eq!(tool_call_id.as_deref(), Some(tool_id));
        assert_eq!(state, &ExecutionCheckpointState::ToolStarted);
        assert!(idempotency_key.contains(tool_id));
        assert!(idempotency_key.contains(&turn_text));
        assert!(idempotency_key.contains(&step_text));
    }

    #[test]
    fn non_boundary_stream_events_do_not_create_execution_checkpoints() {
        let checkpoints = execution_checkpoints_from_agent_event(
            &AgentEvent::TextDelta { delta: "hi".into() },
            None,
            None,
        );
        assert!(checkpoints.is_empty());
    }

    #[test]
    fn failed_tool_started_checkpoint_prevents_execution_gate() {
        let record_dir = tempfile::tempdir().expect("record dir");
        // A directory is intentionally supplied as the record path so the
        // mandatory append fails before the side-effect closure is reached.
        let writer = AgentRecordWriter::from_path(record_dir.path()).expect("writer");
        let mut executed = false;
        let execution = tool_batch::append_tool_execution_checkpoint(
            &writer,
            Some("turn-1"),
            Some("step-1"),
            "call-1",
            ExecutionCheckpointState::ToolStarted,
        )
        .map(|_| {
            executed = true;
        });
        assert!(execution.is_err());
        assert!(!executed, "tool execution must remain gated on persistence");
    }
}
