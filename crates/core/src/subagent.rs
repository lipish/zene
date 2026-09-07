use std::path::Path;
use std::sync::Arc;

use crate::tool_executor::{execute_subagent_tool_batch, SubagentToolBatchDeps};
use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zene_config::ZeneConfig;
use zene_llm::{ChatClient, Message, TokenUsage, ToolCall};
use zene_model_executor::{ChatClientExecutor, ModelExecutor, ModelRequest};
#[cfg(test)]
use zene_model_executor::{ModelResponse, ModelStream};
use zene_sandbox::Sandbox;
use zene_tools::{
    RuntimeScope, SubagentEnv, SubagentProfile, SubagentRunner, ToolCatalog, ToolContext,
    ToolRegistry,
};

use crate::context_config;
use zene_context::{
    compact_message_list_with_chat, estimate_context, should_compact, subagent_compaction_config,
    TokenEstimator,
};
use zene_permission::SharedToolPermission;
use zene_turn::{
    aborted_error, max_turns_notice, ContextAssemblerPort, EventSinkPort, ModelExecutorPort,
    PreparedContext, StepResult, ToolBatchOutcome, ToolExecutorPort, TurnEngine, TurnEnginePorts,
    TurnRequest, TurnRuntime, TurnSessionPort, TurnState,
};

pub struct CoreSubagentRunner {
    config: ZeneConfig,
    broker: Option<zene_permission::SharedApprovalBroker>,
}

impl CoreSubagentRunner {
    pub fn new(config: ZeneConfig) -> Self {
        Self {
            config,
            broker: None,
        }
    }

    pub fn with_broker(mut self, broker: Option<zene_permission::SharedApprovalBroker>) -> Self {
        self.broker = broker;
        self
    }
}

async fn model_executor_from_config(config: &ZeneConfig) -> Result<Arc<dyn ModelExecutor>> {
    let client = ChatClient::from_config(config).await?;
    Ok(Arc::new(ChatClientExecutor::new(Arc::new(client))))
}

#[async_trait]
impl SubagentRunner for CoreSubagentRunner {
    async fn run_subagent(
        &self,
        prompt: &str,
        profile: SubagentProfile,
        cwd: Option<&Path>,
        parent_ctx: &ToolContext,
    ) -> Result<String> {
        let executor = model_executor_from_config(&self.config).await?;
        let sandbox = resolve_subagent_sandbox(&parent_ctx.sandbox, cwd)?;
        let parent_depth = parent_ctx
            .subagent
            .as_ref()
            .map(|env| env.depth)
            .unwrap_or(0);

        run_subagent(
            prompt,
            profile,
            sandbox,
            &self.config,
            executor,
            SubagentOptions {
                cancel: parent_ctx.cancel.as_ref(),
                parent_depth,
                permission: parent_ctx.permission.clone(),
                runner: None,
                broker: self.broker.clone(),
            },
        )
        .await
    }
}

#[derive(Clone, Default)]
pub struct SubagentOptions<'a> {
    pub cancel: Option<&'a CancellationToken>,
    pub parent_depth: u32,
    pub permission: Option<SharedToolPermission>,
    pub runner: Option<Arc<dyn SubagentRunner>>,
    pub broker: Option<zene_permission::SharedApprovalBroker>,
}

pub async fn run_subagent(
    prompt: &str,
    profile: SubagentProfile,
    sandbox: Arc<dyn Sandbox>,
    config: &ZeneConfig,
    model_executor: Arc<dyn ModelExecutor>,
    options: SubagentOptions<'_>,
) -> Result<String> {
    let scope = RuntimeScope::subagent(profile, options.parent_depth)?;
    let runner = options.runner.unwrap_or_else(|| {
        Arc::new(CoreSubagentRunner::new(config.clone()).with_broker(options.broker.clone()))
    });
    let subagent_env = scope.env(runner);
    let mut runtime = SubagentTurnRuntime::new(
        scope,
        sandbox,
        config,
        model_executor,
        subagent_env,
        options.permission,
        options.broker,
    );

    TurnEngine::new(&mut runtime)
        .run(TurnRequest::new(prompt, &(), options.cancel))
        .await
        .map(|outcome| outcome.final_text)
}

/// Direct TurnEngine ports for ephemeral subagents.
///
/// Subagents share the generic turn state machine. Conversation stays in
/// memory; they do not publish the parent runtime's events or checkpoints.
/// Tools come from [`RuntimeScope`]; model calls go through [`ModelExecutor`].
struct SubagentTurnRuntime<'a> {
    sandbox: Arc<dyn Sandbox>,
    config: &'a ZeneConfig,
    model_executor: Arc<dyn ModelExecutor>,
    scope: RuntimeScope,
    subagent_env: SubagentEnv,
    permission: Option<SharedToolPermission>,
    broker: Option<zene_permission::SharedApprovalBroker>,
    tools: Arc<ToolRegistry>,
    messages: Vec<Message>,
    compaction_config: zene_context::CompactionConfig,
    active_turn: Option<TurnState>,
}

impl<'a> SubagentTurnRuntime<'a> {
    fn new(
        scope: RuntimeScope,
        sandbox: Arc<dyn Sandbox>,
        config: &'a ZeneConfig,
        model_executor: Arc<dyn ModelExecutor>,
        subagent_env: SubagentEnv,
        permission: Option<SharedToolPermission>,
        broker: Option<zene_permission::SharedApprovalBroker>,
    ) -> Self {
        let profile = scope
            .subagent_profile()
            .expect("SubagentTurnRuntime requires a subagent RuntimeScope");
        let system_prompt = subagent_system_prompt(profile, sandbox.workdir());
        let tools = Arc::new(scope.tools());
        Self {
            sandbox,
            config,
            model_executor,
            scope,
            subagent_env,
            permission,
            broker,
            tools,
            messages: vec![Message::system(&system_prompt)],
            compaction_config: subagent_compaction_config(
                &context_config::context_compaction_config(&config.compaction),
            ),
            active_turn: None,
        }
    }

    async fn assemble_context(
        &mut self,
        cancel: Option<&CancellationToken>,
    ) -> Result<PreparedContext> {
        if check_cancelled(cancel)? {
            return Err(aborted_error());
        }
        maybe_compact_subagent_messages(
            &mut self.messages,
            self.tools.as_ref(),
            &self.compaction_config,
            &self.config.model,
            Arc::clone(&self.model_executor),
        )
        .await?;
        Ok(PreparedContext {
            messages: self.messages.clone(),
            tools: ToolCatalog::definitions(self.tools.as_ref()),
            context_epoch: None,
            metadata: None,
            estimate_tokens: None,
        })
    }

    async fn execute_tools(
        &mut self,
        tool_calls: &[ToolCall],
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome> {
        let batch = execute_subagent_tool_batch(
            SubagentToolBatchDeps {
                tools: Arc::clone(&self.tools),
                sandbox: Arc::clone(&self.sandbox),
                subagent_env: self.subagent_env.clone(),
                permission: self.permission.clone(),
                broker: self.broker.clone(),
                tool_policy: self.scope.tool_policy,
            },
            tool_calls,
            cancel,
        )
        .await?;
        for message in batch.messages {
            self.messages.push(Message::tool_result_with_error(
                &message.call.id,
                &message.call.name,
                message.content,
                message.is_error,
            ));
        }
        Ok(batch.outcome)
    }

    async fn invoke_model(
        &mut self,
        context: PreparedContext,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult> {
        if check_cancelled(cancel)? {
            return Err(aborted_error());
        }
        let response = self
            .model_executor
            .complete(ModelRequest {
                model: self.config.model.clone(),
                messages: context.messages,
                tools: context.tools,
                stream: false,
                context: None,
                reasoning_effort: self.config.reasoning_effort.clone(),
            })
            .await
            .context("subagent llm step")?;
        let had_tool_calls = response
            .message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
        Ok(StepResult {
            message: response.message,
            usage: response.usage,
            had_tool_calls,
        })
    }
}

impl TurnEnginePorts for SubagentTurnRuntime<'_> {
    type Options = ();
}

#[async_trait]
impl TurnSessionPort<()> for SubagentTurnRuntime<'_> {
    fn max_steps(&self) -> u32 {
        self.config.max_turns
    }

    fn active_turn(&mut self) -> Option<&mut TurnState> {
        self.active_turn.as_mut()
    }

    async fn prepare_turn(&mut self, user_input: &str) -> Result<()> {
        self.active_turn = Some(TurnState::begin());
        self.messages.push(Message::user(user_input));
        Ok(())
    }

    fn inject_steer(&mut self, _options: &()) -> Result<bool> {
        // Ephemeral child scopes disable steer via SessionPolicy.
        Ok(self.scope.session_policy.steer)
    }

    fn push_assistant(&mut self, message: Message) {
        self.messages.push(message);
    }

    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        _options: &(),
    ) -> Result<()> {
        let notice = max_turns_notice(max_steps);
        *final_text = if final_text.trim().is_empty() {
            notice
        } else {
            format!("{final_text}\n\n{notice}")
        };
        self.messages.push(Message::assistant(final_text.clone()));
        Ok(())
    }

    async fn finish_turn(&mut self) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ContextAssemblerPort<()> for SubagentTurnRuntime<'_> {
    async fn prepare_context(
        &mut self,
        _options: &(),
        cancel: Option<&CancellationToken>,
    ) -> Result<PreparedContext> {
        self.assemble_context(cancel).await
    }
}

#[async_trait]
impl ModelExecutorPort<()> for SubagentTurnRuntime<'_> {
    async fn run_model(
        &mut self,
        context: PreparedContext,
        _options: &(),
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult> {
        self.invoke_model(context, cancel).await
    }

    async fn on_step_usage(&mut self, _usage: &TokenUsage, _options: &()) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ToolExecutorPort<()> for SubagentTurnRuntime<'_> {
    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        _options: &(),
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome> {
        self.execute_tools(tool_calls, cancel).await
    }
}

impl EventSinkPort<()> for SubagentTurnRuntime<'_> {
    fn on_step_begin(
        &self,
        _turn_id: zene_turn::TurnId,
        _step_id: zene_turn::StepId,
        _step: u32,
        _options: &(),
    ) {
    }
}

#[async_trait]
impl TurnRuntime for SubagentTurnRuntime<'_> {
    type Options = ();

    fn max_steps(&self) -> u32 {
        self.config.max_turns
    }

    fn active_turn(&mut self) -> Option<&mut TurnState> {
        self.active_turn.as_mut()
    }

    fn on_step_begin(
        &self,
        _turn_id: zene_turn::TurnId,
        _step_id: zene_turn::StepId,
        _step: u32,
        _options: &Self::Options,
    ) {
    }

    async fn prepare_turn(&mut self, user_input: &str) -> Result<()> {
        self.active_turn = Some(TurnState::begin());
        self.messages.push(Message::user(user_input));
        Ok(())
    }

    async fn run_step(
        &mut self,
        _options: &Self::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult> {
        let context = self.assemble_context(cancel).await?;
        self.invoke_model(context, cancel).await
    }

    async fn on_step_usage(&mut self, _usage: &TokenUsage, _options: &Self::Options) -> Result<()> {
        Ok(())
    }

    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        _options: &Self::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome> {
        self.execute_tools(tool_calls, cancel).await
    }

    fn inject_steer(&mut self, _options: &Self::Options) -> Result<bool> {
        Ok(self.scope.session_policy.steer)
    }

    fn push_assistant(&mut self, message: Message) {
        self.messages.push(message);
    }

    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        _options: &Self::Options,
    ) -> Result<()> {
        let notice = max_turns_notice(max_steps);
        *final_text = if final_text.trim().is_empty() {
            notice
        } else {
            format!("{final_text}\n\n{notice}")
        };
        self.messages.push(Message::assistant(final_text.clone()));
        Ok(())
    }

    async fn finish_turn(&mut self) -> Result<()> {
        Ok(())
    }
}

fn subagent_system_prompt(profile: SubagentProfile, workdir: &Path) -> String {
    let workdir = workdir
        .canonicalize()
        .unwrap_or_else(|_| workdir.to_path_buf());

    match profile {
        SubagentProfile::Explore => format!(
            "You are an explore subagent for Zene. Investigate the codebase using read-only tools (Read, Grep, Glob). \
             Report findings concisely. Working directory: `{}`.",
            workdir.display()
        ),
        SubagentProfile::Coder => format!(
            "You are a coder subagent for Zene. Read and modify code as requested using the available tools. \
             Working directory: `{}`.",
            workdir.display()
        ),
    }
}

fn resolve_subagent_sandbox(
    parent: &Arc<dyn Sandbox>,
    cwd: Option<&Path>,
) -> Result<Arc<dyn Sandbox>> {
    match cwd {
        None => Ok(Arc::clone(parent)),
        Some(path) => {
            let resolved = parent.resolve(path.to_str().unwrap_or(""))?;
            if !resolved.is_dir() {
                anyhow::bail!("Task cwd is not a directory: {}", resolved.display());
            }
            parent.scoped_to(resolved)
        }
    }
}

fn check_cancelled(cancel: Option<&CancellationToken>) -> Result<bool> {
    Ok(cancel.is_some_and(CancellationToken::is_cancelled))
}

async fn maybe_compact_subagent_messages(
    messages: &mut Vec<Message>,
    tools: &ToolRegistry,
    compaction_config: &zene_context::CompactionConfig,
    model: &str,
    model_executor: Arc<dyn ModelExecutor>,
) -> Result<()> {
    let tool_defs = tools.definitions();
    let estimator = TokenEstimator::default();
    let estimated = estimate_context(messages, &tool_defs, &estimator) as u32;
    if !should_compact(estimated, compaction_config) {
        return Ok(());
    }

    if compact_message_list_with_chat(
        messages,
        model,
        compaction_config,
        "subagent_token_threshold",
        &tool_defs,
        &estimator,
        |request| {
            let model_executor = Arc::clone(&model_executor);
            async move { model_executor.complete(request).await }
        },
    )
    .await?
    .is_some()
    {
        ensure_subagent_system_message(messages);
    }

    Ok(())
}

fn ensure_subagent_system_message(messages: &mut Vec<Message>) {
    if messages
        .first()
        .is_some_and(|m| m.role == zene_llm::Role::System)
    {
        return;
    }
    if let Some(system) = messages.iter().find(|m| m.role == zene_llm::Role::System) {
        let system = system.clone();
        messages.retain(|m| m.role != zene_llm::Role::System);
        messages.insert(0, system);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use parking_lot::Mutex;

    use tempfile::tempdir;
    use zene_llm::ToolCall;
    use zene_permission::{PermissionGate, PermissionMode, PromptChoice, SharedToolPermission};
    use zene_sandbox::LocalSandbox;
    use zene_tools::{default_builtin_tools, DEFAULT_SUBAGENT_MAX_DEPTH};

    fn test_permission_deny() -> SharedToolPermission {
        Arc::new(Mutex::new(PermissionGate::with_prompter(
            PermissionMode::Manual,
            Box::new(|_tool, _args| Ok(PromptChoice::Deny)),
        )))
    }

    type FirstCallHook = Box<dyn Fn(&ModelRequest) + Send + Sync>;

    struct ScriptedBackend {
        responses: Vec<ModelResponse>,
        calls: AtomicUsize,
        on_first_call: Option<FirstCallHook>,
    }

    impl ScriptedBackend {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses,
                calls: AtomicUsize::new(0),
                on_first_call: None,
            }
        }

        fn with_first_call_check(
            responses: Vec<ModelResponse>,
            check: impl Fn(&ModelRequest) + Send + Sync + 'static,
        ) -> Self {
            Self {
                responses,
                calls: AtomicUsize::new(0),
                on_first_call: Some(Box::new(check)),
            }
        }
    }

    #[async_trait]
    impl ModelExecutor for ScriptedBackend {
        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            let response = self
                .responses
                .get(idx)
                .cloned()
                .with_context(|| format!("no scripted response for call {idx}"))?;

            if idx == 0 {
                if let Some(check) = &self.on_first_call {
                    check(&request);
                }
            }

            Ok(response)
        }

        async fn stream(&self, _request: ModelRequest) -> Result<ModelStream> {
            Ok(Box::pin(futures::stream::iter([Ok(
                zene_llm::StreamEvent::Done { usage: None },
            )])))
        }
    }

    struct RecordingRunner {
        config: ZeneConfig,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl SubagentRunner for RecordingRunner {
        async fn run_subagent(
            &self,
            prompt: &str,
            profile: SubagentProfile,
            cwd: Option<&Path>,
            parent_ctx: &ToolContext,
        ) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let executor = model_executor_from_config(&self.config).await?;
            let sandbox = resolve_subagent_sandbox(&parent_ctx.sandbox, cwd)?;
            let parent_depth = parent_ctx
                .subagent
                .as_ref()
                .map(|env| env.depth)
                .unwrap_or(0);
            run_subagent(
                prompt,
                profile,
                sandbox,
                &self.config,
                executor,
                SubagentOptions {
                    cancel: parent_ctx.cancel.as_ref(),
                    parent_depth,
                    permission: parent_ctx.permission.clone(),
                    ..Default::default()
                },
            )
            .await
        }
    }

    #[tokio::test]
    async fn explore_subagent_lists_files_via_glob_without_write() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("alpha.txt"), "a")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("beta.txt"), "b")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("notes.md"), "n")
            .await
            .unwrap();

        let sandbox = zene_sandbox::into_arc(LocalSandbox::new(dir.path()));
        let config = ZeneConfig::default();

        let backend = ScriptedBackend::with_first_call_check(
            vec![
                ModelResponse {
                    message: Message::assistant_with_tools(
                        None,
                        vec![ToolCall {
                            id: "call_glob".to_string(),
                            name: "Glob".to_string(),
                            arguments: r#"{"pattern":"**/*.txt"}"#.to_string(),
                        }],
                    ),
                    usage: None,
                },
                ModelResponse {
                    message: Message::assistant("Found alpha.txt and beta.txt"),
                    usage: None,
                },
            ],
            |request| {
                let tool_names: Vec<_> = request
                    .tools
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect();
                assert!(tool_names.contains(&"Glob"));
                assert!(!tool_names.contains(&"Write"));
                assert!(!tool_names.contains(&"Task"));
                assert!(
                    request.messages.iter().any(|message| {
                        message.content.as_deref() == Some("List all .txt files in the workspace")
                    }),
                    "subagent model must consume PreparedContext messages"
                );
            },
        );

        let result = run_subagent(
            "List all .txt files in the workspace",
            SubagentProfile::Explore,
            Arc::clone(&sandbox),
            &config,
            Arc::new(backend),
            SubagentOptions::default(),
        )
        .await
        .expect("subagent should complete");

        let explore_tools: Vec<_> = RuntimeScope::subagent(SubagentProfile::Explore, 0)
            .expect("explore scope")
            .definitions()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(explore_tools.contains(&"Glob".to_string()));
        assert!(!explore_tools.contains(&"Write".to_string()));
        assert!(!explore_tools.contains(&"Task".to_string()));

        assert!(result.contains("alpha.txt"));
        assert!(result.contains("beta.txt"));
        assert!(!result.contains("notes.md"));

        let write_attempt = RuntimeScope::subagent(SubagentProfile::Explore, 0)
            .expect("explore scope")
            .tools()
            .execute(
                "Write",
                r#"{"path":"blocked.txt","content":"nope"}"#,
                &ToolContext::without_subagent(Arc::clone(&sandbox)),
            )
            .await;
        assert!(write_attempt.is_err());
        assert!(write_attempt
            .unwrap_err()
            .to_string()
            .contains("unknown tool"));
    }

    #[tokio::test]
    async fn subagent_uses_shared_turn_engine_step_budget() {
        let dir = tempdir().unwrap();
        let sandbox = zene_sandbox::into_arc(LocalSandbox::new(dir.path()));
        let config = ZeneConfig {
            max_turns: 1,
            ..Default::default()
        };
        let backend = ScriptedBackend::new(vec![ModelResponse {
            message: Message::assistant_with_tools(
                None,
                vec![ToolCall {
                    id: "call_glob".into(),
                    name: "Glob".into(),
                    arguments: r#"{"pattern":"**/*.rs"}"#.into(),
                }],
            ),
            usage: None,
        }]);

        let result = run_subagent(
            "Inspect Rust files",
            SubagentProfile::Explore,
            sandbox,
            &config,
            Arc::new(backend),
            SubagentOptions::default(),
        )
        .await
        .expect("incomplete subagent should return a notice");
        assert!(result.contains("Reached max_turns (1)"));
    }

    #[tokio::test]
    async fn task_tool_rejects_nested_subagent_at_max_depth() {
        let dir = tempdir().unwrap();
        let sandbox = zene_sandbox::into_arc(LocalSandbox::new(dir.path()));
        let config = ZeneConfig::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = Arc::new(RecordingRunner {
            config,
            calls: Arc::clone(&calls),
        });

        let env = SubagentEnv {
            depth: DEFAULT_SUBAGENT_MAX_DEPTH,
            max_depth: DEFAULT_SUBAGENT_MAX_DEPTH,
            runner,
        };
        let ctx = ToolContext {
            sandbox: Arc::clone(&sandbox),
            cancel: None,
            subagent: Some(env),
            permission: None,
            plan_mode: None,
            todos: None,
            ask_user: None,
            background: None,
        };

        let result = default_builtin_tools()
            .execute("Task", r#"{"prompt":"nested","agent":"explore"}"#, &ctx)
            .await
            .expect("task execution");

        assert!(result.is_error);
        assert!(result.content.contains("nesting limit"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn coder_subagent_manual_mode_rejects_write_without_approval() {
        let dir = tempdir().unwrap();
        let sandbox = zene_sandbox::into_arc(LocalSandbox::new(dir.path()));
        let config = ZeneConfig::default();
        let permission = test_permission_deny();

        let backend = ScriptedBackend::new(vec![
            ModelResponse {
                message: Message::assistant_with_tools(
                    None,
                    vec![ToolCall {
                        id: "call_write".to_string(),
                        name: "Write".to_string(),
                        arguments: r#"{"path":"secret.txt","content":"nope"}"#.to_string(),
                    }],
                ),
                usage: None,
            },
            ModelResponse {
                message: Message::assistant("Write was denied"),
                usage: None,
            },
        ]);

        let result = run_subagent(
            "Write secret.txt",
            SubagentProfile::Coder,
            Arc::clone(&sandbox),
            &config,
            Arc::new(backend),
            SubagentOptions {
                permission: Some(permission),
                ..Default::default()
            },
        )
        .await
        .expect("subagent should complete after denial");

        assert!(result.contains("denied") || result.contains("Denied"));
        assert!(!dir.path().join("secret.txt").exists());
    }
}
