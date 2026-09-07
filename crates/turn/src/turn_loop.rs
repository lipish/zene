use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use zene_llm::{ContextMetadata, Message, TokenUsage, ToolCall, ToolDefinition};

use crate::state::{aborted_error, is_cancelled, StepId, TurnId, TurnState};

/// Outcome of one LLM step within a turn.
pub struct StepResult {
    pub message: Message,
    pub usage: Option<TokenUsage>,
    pub had_tool_calls: bool,
}

/// Controls whether a completed tool batch should cause another model call.
///
/// `Continue` preserves the normal LLM → tools → LLM loop. `Terminate` is the
/// escape hatch for executors and extensions that have produced the terminal
/// result for a turn and must not trigger an automatic follow-up model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolBatchOutcome {
    Continue,
    Terminate,
}

/// Stable lifecycle result returned by [`TurnEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStatus {
    /// The model produced a final answer.
    Completed,
    /// A tool batch produced the terminal result for the turn.
    Terminated,
    /// The step budget was exhausted before a final answer.
    Incomplete,
}

/// Result of one turn-engine execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    pub final_text: String,
    pub status: TurnStatus,
    pub steps: u32,
}

/// Input boundary for one turn-engine execution.
pub struct TurnRequest<'a, O> {
    pub user_input: &'a str,
    pub options: &'a O,
    pub cancel: Option<&'a CancellationToken>,
}

impl<'a, O> TurnRequest<'a, O> {
    pub fn new(user_input: &'a str, options: &'a O, cancel: Option<&'a CancellationToken>) -> Self {
        Self {
            user_input,
            options,
            cancel,
        }
    }
}

/// Runtime hooks consumed by the turn engine.
///
/// This is intentionally free of `Agent`, ACP, Cloud, and provider types. The
/// existing name is retained as a compatibility contract for callers that
/// implement the turn hooks directly.
#[async_trait]
pub trait TurnRuntime {
    type Options: Send + Sync;

    fn max_steps(&self) -> u32;
    fn active_turn(&mut self) -> Option<&mut TurnState>;

    fn on_step_begin(&self, turn_id: TurnId, step_id: StepId, step: u32, options: &Self::Options);

    async fn prepare_turn(&mut self, user_input: &str) -> Result<()>;
    async fn run_step(
        &mut self,
        options: &Self::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult>;
    async fn on_step_usage(&mut self, usage: &TokenUsage, options: &Self::Options) -> Result<()>;
    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        options: &Self::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome>;
    fn inject_steer(&mut self, options: &Self::Options) -> Result<bool>;
    fn inject_follow_up(&mut self, _options: &Self::Options) -> Result<bool> {
        Ok(false)
    }
    fn push_assistant(&mut self, message: Message);
    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        options: &Self::Options,
    ) -> Result<()>;
    async fn finish_turn(&mut self) -> Result<()>;
}

/// Session/state capabilities consumed by [`TurnEngine`].
#[async_trait]
pub trait TurnSessionPort<O>: Send {
    fn max_steps(&self) -> u32;
    fn active_turn(&mut self) -> Option<&mut TurnState>;
    async fn prepare_turn(&mut self, user_input: &str) -> Result<()>;
    fn inject_steer(&mut self, options: &O) -> Result<bool>;
    fn inject_follow_up(&mut self, _options: &O) -> Result<bool> {
        Ok(false)
    }
    fn push_assistant(&mut self, message: Message);
    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        options: &O,
    ) -> Result<()>;
    async fn finish_turn(&mut self) -> Result<()>;
}

/// Context projected for one model invocation.
///
/// TurnEngine passes this value from [`ContextAssemblerPort`] to
/// [`ModelExecutorPort`]. Default-path ports must consume these fields rather
/// than re-assembling context inside `run_model`.
#[derive(Debug, Clone, Default)]
pub struct PreparedContext {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub context_epoch: Option<u64>,
    pub metadata: Option<ContextMetadata>,
    pub estimate_tokens: Option<u32>,
}

/// Context projection capabilities consumed by [`TurnEngine`].
#[async_trait]
pub trait ContextAssemblerPort<O>: Send {
    async fn prepare_context(
        &mut self,
        options: &O,
        cancel: Option<&CancellationToken>,
    ) -> Result<PreparedContext>;
}

/// Model invocation capabilities consumed by [`TurnEngine`].
#[async_trait]
pub trait ModelExecutorPort<O>: Send {
    async fn run_model(
        &mut self,
        context: PreparedContext,
        options: &O,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult>;
    async fn on_step_usage(&mut self, usage: &TokenUsage, options: &O) -> Result<()>;
}

/// Tool-batch capabilities consumed by [`TurnEngine`].
#[async_trait]
pub trait ToolExecutorPort<O>: Send {
    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        options: &O,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome>;
}

/// Event publication capabilities consumed by [`TurnEngine`].
pub trait EventSinkPort<O>: Send {
    fn on_step_begin(&self, turn_id: TurnId, step_id: StepId, step: u32, options: &O);
}

/// Stable bundle of ports required by the turn state machine.
pub trait TurnEnginePorts:
    TurnSessionPort<Self::Options>
    + ContextAssemblerPort<Self::Options>
    + ModelExecutorPort<Self::Options>
    + ToolExecutorPort<Self::Options>
    + EventSinkPort<Self::Options>
{
    type Options: Send + Sync;
}

/// Compatibility adapter for the pre-Wave 5 [`TurnRuntime`] contract.
pub struct LegacyTurnPorts<'a, R: TurnRuntime> {
    runtime: &'a mut R,
}

impl<'a, R: TurnRuntime> LegacyTurnPorts<'a, R> {
    pub fn new(runtime: &'a mut R) -> Self {
        Self { runtime }
    }
}

impl<R: TurnRuntime + Send> TurnEnginePorts for LegacyTurnPorts<'_, R> {
    type Options = R::Options;
}

#[async_trait]
impl<R: TurnRuntime + Send> TurnSessionPort<R::Options> for LegacyTurnPorts<'_, R> {
    fn max_steps(&self) -> u32 {
        self.runtime.max_steps()
    }

    fn active_turn(&mut self) -> Option<&mut TurnState> {
        self.runtime.active_turn()
    }

    async fn prepare_turn(&mut self, user_input: &str) -> Result<()> {
        self.runtime.prepare_turn(user_input).await
    }

    fn inject_steer(&mut self, options: &R::Options) -> Result<bool> {
        self.runtime.inject_steer(options)
    }

    fn inject_follow_up(&mut self, options: &R::Options) -> Result<bool> {
        self.runtime.inject_follow_up(options)
    }

    fn push_assistant(&mut self, message: Message) {
        self.runtime.push_assistant(message);
    }

    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        options: &R::Options,
    ) -> Result<()> {
        self.runtime
            .on_incomplete_turn(max_steps, final_text, options)
    }

    async fn finish_turn(&mut self) -> Result<()> {
        self.runtime.finish_turn().await
    }
}

#[async_trait]
impl<R: TurnRuntime + Send> ContextAssemblerPort<R::Options> for LegacyTurnPorts<'_, R> {
    async fn prepare_context(
        &mut self,
        _options: &R::Options,
        _cancel: Option<&CancellationToken>,
    ) -> Result<PreparedContext> {
        // Legacy runtimes prepare context inside `run_step`; this no-op keeps
        // the migration adapter behaviorally compatible.
        Ok(PreparedContext::default())
    }
}

#[async_trait]
impl<R: TurnRuntime + Send> ModelExecutorPort<R::Options> for LegacyTurnPorts<'_, R> {
    async fn run_model(
        &mut self,
        _context: PreparedContext,
        options: &R::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult> {
        self.runtime.run_step(options, cancel).await
    }

    async fn on_step_usage(&mut self, usage: &TokenUsage, options: &R::Options) -> Result<()> {
        self.runtime.on_step_usage(usage, options).await
    }
}

#[async_trait]
impl<R: TurnRuntime + Send> ToolExecutorPort<R::Options> for LegacyTurnPorts<'_, R> {
    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        options: &R::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome> {
        self.runtime.run_tools(tool_calls, options, cancel).await
    }
}

impl<R: TurnRuntime + Send> EventSinkPort<R::Options> for LegacyTurnPorts<'_, R> {
    fn on_step_begin(&self, turn_id: TurnId, step_id: StepId, step: u32, options: &R::Options) {
        self.runtime.on_step_begin(turn_id, step_id, step, options);
    }
}

/// Generic multi-step turn state machine.
///
/// The engine owns orchestration only. Context, model, tool, session, and
/// event behavior is supplied through explicit ports, keeping the state
/// machine independent from `zene-core::Agent` and all transports.
pub struct TurnEngine<'a, P>
where
    P: TurnEnginePorts,
{
    ports: &'a mut P,
}

impl<'a, P> TurnEngine<'a, P>
where
    P: TurnEnginePorts,
{
    pub fn new(ports: &'a mut P) -> Self {
        Self { ports }
    }

    pub async fn run<'r>(&mut self, request: TurnRequest<'r, P::Options>) -> Result<TurnOutcome> {
        let ports = &mut *self.ports;
        ports.prepare_turn(request.user_input).await?;

        let mut final_text = String::new();
        let max_steps = ports.max_steps();
        let mut status = None;
        let mut steps_done = 0u32;

        loop {
            if max_steps > 0 && steps_done >= max_steps {
                break;
            }
            steps_done = steps_done.saturating_add(1);

            if is_cancelled(request.cancel) {
                return Err(aborted_error());
            }

            let (turn_id, step_id, step_num) = {
                let turn = ports
                    .active_turn()
                    .expect("active turn during TurnEngine::run");
                let step_id = turn.next_step_id();
                (turn.turn_id, step_id, turn.step)
            };

            debug!(
                turn_id = %turn_id,
                step_id = %step_id,
                step = step_num,
                "step_begin"
            );
            ports.on_step_begin(turn_id, step_id, step_num, request.options);

            let context = ports
                .prepare_context(request.options, request.cancel)
                .await?;
            let step_result = ports
                .run_model(context, request.options, request.cancel)
                .await;
            debug!(
                turn_id = %turn_id,
                step_id = %step_id,
                step = step_num,
                ok = step_result.is_ok(),
                "step_end"
            );
            let StepResult {
                message: assistant_message,
                usage,
                had_tool_calls,
            } = step_result?;

            if let Some(usage) = &usage {
                ports.on_step_usage(usage, request.options).await?;
            }

            if had_tool_calls {
                if let Some(tool_calls) = assistant_message.tool_calls.clone() {
                    ports.push_assistant(assistant_message);
                    let tool_outcome = ports
                        .run_tools(&tool_calls, request.options, request.cancel)
                        .await?;
                    if tool_outcome == ToolBatchOutcome::Terminate {
                        status = Some(TurnStatus::Terminated);
                        break;
                    }
                    if ports.inject_steer(request.options)? {
                        continue;
                    }
                    continue;
                }
            }

            ports.push_assistant(assistant_message.clone());
            if ports.inject_steer(request.options)? {
                continue;
            }

            final_text = assistant_message.content.unwrap_or_default();
            if ports.inject_follow_up(request.options)? {
                continue;
            }
            status = Some(TurnStatus::Completed);
            break;
        }

        let status = match status {
            Some(status) => status,
            None => {
                ports.on_incomplete_turn(max_steps, &mut final_text, request.options)?;
                TurnStatus::Incomplete
            }
        };
        let steps = ports
            .active_turn()
            .map(|turn| turn.step)
            .unwrap_or(steps_done);

        ports.finish_turn().await?;
        Ok(TurnOutcome {
            final_text,
            status,
            steps,
        })
    }
}

/// Backward-compatible string-returning facade for the generic turn engine.
pub async fn run_turn_loop<R: TurnRuntime + Send>(
    runtime: &mut R,
    user_input: &str,
    options: &R::Options,
    cancel: Option<&CancellationToken>,
) -> Result<String> {
    let mut ports = LegacyTurnPorts::new(runtime);
    TurnEngine::new(&mut ports)
        .run(TurnRequest::new(user_input, options, cancel))
        .await
        .map(|outcome| outcome.final_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn block_on<F: Future>(mut future: F) -> F::Output {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut context = Context::from_waker(&waker);
        let mut future = unsafe { Pin::new_unchecked(&mut future) };
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    struct FakeRuntime {
        active: Option<TurnState>,
        model_calls: Arc<AtomicUsize>,
        tool_outcome: ToolBatchOutcome,
        max_steps: u32,
        steer_calls: Arc<AtomicUsize>,
    }

    struct DirectPorts {
        active: Option<TurnState>,
        model_calls: Arc<AtomicUsize>,
        max_steps: u32,
        seen_user: Arc<std::sync::Mutex<Option<String>>>,
    }

    impl TurnEnginePorts for DirectPorts {
        type Options = ();
    }

    #[async_trait]
    impl ContextAssemblerPort<()> for DirectPorts {
        async fn prepare_context(
            &mut self,
            _options: &(),
            _cancel: Option<&CancellationToken>,
        ) -> Result<PreparedContext> {
            Ok(PreparedContext {
                messages: vec![Message::user("from-assembler")],
                tools: Vec::new(),
                context_epoch: Some(7),
                metadata: None,
                estimate_tokens: Some(12),
            })
        }
    }

    #[async_trait]
    impl TurnSessionPort<()> for DirectPorts {
        fn max_steps(&self) -> u32 {
            self.max_steps
        }

        fn active_turn(&mut self) -> Option<&mut TurnState> {
            self.active.as_mut()
        }

        async fn prepare_turn(&mut self, _user_input: &str) -> Result<()> {
            self.active = Some(TurnState::begin());
            Ok(())
        }

        fn inject_steer(&mut self, _options: &()) -> Result<bool> {
            Ok(false)
        }

        fn push_assistant(&mut self, _message: Message) {}

        fn on_incomplete_turn(
            &mut self,
            _max_steps: u32,
            _final_text: &mut String,
            _options: &(),
        ) -> Result<()> {
            Ok(())
        }

        async fn finish_turn(&mut self) -> Result<()> {
            self.active = None;
            Ok(())
        }
    }

    #[async_trait]
    impl ModelExecutorPort<()> for DirectPorts {
        async fn run_model(
            &mut self,
            context: PreparedContext,
            _options: &(),
            _cancel: Option<&CancellationToken>,
        ) -> Result<StepResult> {
            let call = self.model_calls.fetch_add(1, Ordering::SeqCst);
            *self.seen_user.lock().expect("seen_user") = context
                .messages
                .first()
                .and_then(|message| message.content.clone());
            Ok(if call == 0 {
                StepResult {
                    message: Message::assistant("direct"),
                    usage: None,
                    had_tool_calls: false,
                }
            } else {
                StepResult {
                    message: Message::assistant("unexpected"),
                    usage: None,
                    had_tool_calls: false,
                }
            })
        }

        async fn on_step_usage(&mut self, _usage: &TokenUsage, _options: &()) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl ToolExecutorPort<()> for DirectPorts {
        async fn run_tools(
            &mut self,
            _tool_calls: &[ToolCall],
            _options: &(),
            _cancel: Option<&CancellationToken>,
        ) -> Result<ToolBatchOutcome> {
            Ok(ToolBatchOutcome::Continue)
        }
    }

    impl EventSinkPort<()> for DirectPorts {
        fn on_step_begin(&self, _turn_id: TurnId, _step_id: StepId, _step: u32, _options: &()) {}
    }

    #[async_trait]
    impl TurnRuntime for FakeRuntime {
        type Options = ();

        fn max_steps(&self) -> u32 {
            self.max_steps
        }

        fn active_turn(&mut self) -> Option<&mut TurnState> {
            self.active.as_mut()
        }

        fn on_step_begin(
            &self,
            _turn_id: TurnId,
            _step_id: StepId,
            _step: u32,
            _options: &Self::Options,
        ) {
        }

        async fn prepare_turn(&mut self, _user_input: &str) -> Result<()> {
            self.active = Some(TurnState::begin());
            Ok(())
        }

        async fn run_step(
            &mut self,
            _options: &Self::Options,
            _cancel: Option<&CancellationToken>,
        ) -> Result<StepResult> {
            let call = self.model_calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                Ok(StepResult {
                    message: Message::assistant_with_tools(
                        None,
                        vec![ToolCall {
                            id: "call_1".into(),
                            name: "Terminal".into(),
                            arguments: "{}".into(),
                        }],
                    ),
                    usage: None,
                    had_tool_calls: true,
                })
            } else {
                Ok(StepResult {
                    message: Message::assistant("final"),
                    usage: None,
                    had_tool_calls: false,
                })
            }
        }

        async fn on_step_usage(
            &mut self,
            _usage: &TokenUsage,
            _options: &Self::Options,
        ) -> Result<()> {
            Ok(())
        }

        async fn run_tools(
            &mut self,
            _tool_calls: &[ToolCall],
            _options: &Self::Options,
            _cancel: Option<&CancellationToken>,
        ) -> Result<ToolBatchOutcome> {
            Ok(self.tool_outcome)
        }

        fn inject_steer(&mut self, _options: &Self::Options) -> Result<bool> {
            self.steer_calls.fetch_add(1, Ordering::SeqCst);
            Ok(false)
        }

        fn push_assistant(&mut self, _message: Message) {}

        fn on_incomplete_turn(
            &mut self,
            _max_steps: u32,
            _final_text: &mut String,
            _options: &Self::Options,
        ) -> Result<()> {
            Ok(())
        }

        async fn finish_turn(&mut self) -> Result<()> {
            self.active = None;
            Ok(())
        }
    }

    #[test]
    fn engine_accepts_direct_ports_without_legacy_runtime() {
        let model_calls = Arc::new(AtomicUsize::new(0));
        let seen_user = Arc::new(std::sync::Mutex::new(None));
        let mut ports = DirectPorts {
            active: None,
            model_calls: Arc::clone(&model_calls),
            max_steps: 4,
            seen_user: Arc::clone(&seen_user),
        };

        let outcome =
            block_on(TurnEngine::new(&mut ports).run(TurnRequest::new("prompt", &(), None)))
                .expect("turn completes");
        assert_eq!(outcome.final_text, "direct");
        assert_eq!(outcome.status, TurnStatus::Completed);
        assert_eq!(outcome.steps, 1);
        assert_eq!(model_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            seen_user.lock().expect("seen_user").as_deref(),
            Some("from-assembler")
        );
    }

    #[test]
    fn terminate_tool_batch_skips_follow_up_model_call() {
        let model_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = FakeRuntime {
            active: None,
            model_calls: Arc::clone(&model_calls),
            tool_outcome: ToolBatchOutcome::Terminate,
            max_steps: 4,
            steer_calls: Arc::new(AtomicUsize::new(0)),
        };

        let result =
            block_on(run_turn_loop(&mut runtime, "prompt", &(), None)).expect("turn completes");
        assert_eq!(result, "");
        assert_eq!(model_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn terminate_tool_batch_does_not_drain_pending_steer() {
        // Terminate ends the turn immediately, before the steer/follow-up drain
        // point that normally runs after a tool batch. Pending steer input is
        // left queued for the next turn rather than being injected mid-turn.
        let model_calls = Arc::new(AtomicUsize::new(0));
        let steer_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = FakeRuntime {
            active: None,
            model_calls,
            tool_outcome: ToolBatchOutcome::Terminate,
            max_steps: 4,
            steer_calls: Arc::clone(&steer_calls),
        };

        let outcome = block_on(
            TurnEngine::new(&mut LegacyTurnPorts::new(&mut runtime)).run(TurnRequest::new(
                "prompt",
                &(),
                None,
            )),
        )
        .expect("turn completes");
        assert_eq!(outcome.status, TurnStatus::Terminated);
        assert_eq!(steer_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn continue_tool_batch_preserves_follow_up_model_call() {
        let model_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = FakeRuntime {
            active: None,
            model_calls: Arc::clone(&model_calls),
            tool_outcome: ToolBatchOutcome::Continue,
            max_steps: 4,
            steer_calls: Arc::new(AtomicUsize::new(0)),
        };

        let outcome = block_on(
            TurnEngine::new(&mut LegacyTurnPorts::new(&mut runtime)).run(TurnRequest::new(
                "prompt",
                &(),
                None,
            )),
        )
        .expect("turn completes");
        assert_eq!(outcome.final_text, "final");
        assert_eq!(outcome.status, TurnStatus::Completed);
        assert_eq!(outcome.steps, 2);
        assert_eq!(model_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn engine_reports_terminated_status() {
        let model_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = FakeRuntime {
            active: None,
            model_calls,
            tool_outcome: ToolBatchOutcome::Terminate,
            max_steps: 4,
            steer_calls: Arc::new(AtomicUsize::new(0)),
        };

        let outcome = block_on(
            TurnEngine::new(&mut LegacyTurnPorts::new(&mut runtime)).run(TurnRequest::new(
                "prompt",
                &(),
                None,
            )),
        )
        .expect("turn completes");
        assert_eq!(outcome.status, TurnStatus::Terminated);
        assert_eq!(outcome.steps, 1);
    }

    #[test]
    fn engine_reports_incomplete_status_when_step_budget_is_exhausted() {
        let model_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = FakeRuntime {
            active: None,
            model_calls,
            tool_outcome: ToolBatchOutcome::Continue,
            max_steps: 1,
            steer_calls: Arc::new(AtomicUsize::new(0)),
        };

        let outcome = block_on(
            TurnEngine::new(&mut LegacyTurnPorts::new(&mut runtime)).run(TurnRequest::new(
                "prompt",
                &(),
                None,
            )),
        )
        .expect("turn completes");
        assert_eq!(outcome.status, TurnStatus::Incomplete);
        assert_eq!(outcome.steps, 1);
    }
}
