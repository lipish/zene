//! [`TurnRuntime`] implementation for [`Agent`](crate::Agent).

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zene_llm::{TokenUsage, ToolCall};
use zene_turn::{
    ContextAssemblerPort, EventSinkPort, ModelExecutorPort, PreparedContext, StepResult,
    ToolBatchOutcome, ToolExecutorPort, TurnEnginePorts, TurnRuntime, TurnSessionPort,
};

use crate::events::{emit_event, AgentEvent};
use crate::turn_session;
use crate::Agent;

/// Native turn-engine ports for the primary agent runtime.
///
/// Context assembly and model invocation are separate port calls. The engine
/// passes [`PreparedContext`] from assembler to executor; `Agent` no longer
/// re-assembles context inside the model step.
pub(super) struct AgentTurnPorts<'a> {
    agent: &'a mut Agent,
}

impl<'a> AgentTurnPorts<'a> {
    pub(super) fn new(agent: &'a mut Agent) -> Self {
        Self { agent }
    }
}

impl TurnEnginePorts for AgentTurnPorts<'_> {
    type Options = crate::PromptOptions;
}

#[async_trait]
impl TurnSessionPort<crate::PromptOptions> for AgentTurnPorts<'_> {
    fn max_steps(&self) -> u32 {
        <Agent as TurnRuntime>::max_steps(self.agent)
    }

    fn active_turn(&mut self) -> Option<&mut zene_turn::TurnState> {
        <Agent as TurnRuntime>::active_turn(self.agent)
    }

    async fn prepare_turn(&mut self, user_input: &str) -> Result<(), anyhow::Error> {
        <Agent as TurnRuntime>::prepare_turn(self.agent, user_input).await
    }

    fn inject_steer(&mut self, options: &crate::PromptOptions) -> Result<bool, anyhow::Error> {
        <Agent as TurnRuntime>::inject_steer(self.agent, options)
    }

    fn inject_follow_up(&mut self, options: &crate::PromptOptions) -> Result<bool, anyhow::Error> {
        self.agent.inject_pending_follow_up(options)
    }

    fn push_assistant(&mut self, message: zene_llm::Message) {
        <Agent as TurnRuntime>::push_assistant(self.agent, message);
    }

    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        options: &crate::PromptOptions,
    ) -> Result<(), anyhow::Error> {
        <Agent as TurnRuntime>::on_incomplete_turn(self.agent, max_steps, final_text, options)
    }

    async fn finish_turn(&mut self) -> Result<(), anyhow::Error> {
        <Agent as TurnRuntime>::finish_turn(self.agent).await
    }
}

#[async_trait]
impl ContextAssemblerPort<crate::PromptOptions> for AgentTurnPorts<'_> {
    async fn prepare_context(
        &mut self,
        options: &crate::PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<PreparedContext, anyhow::Error> {
        self.agent.record_step_started()?;
        self.agent.prepare_step_context(options, cancel).await
    }
}

#[async_trait]
impl ModelExecutorPort<crate::PromptOptions> for AgentTurnPorts<'_> {
    async fn run_model(
        &mut self,
        context: PreparedContext,
        options: &crate::PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult, anyhow::Error> {
        self.agent.invoke_model(context, options, cancel).await
    }

    async fn on_step_usage(
        &mut self,
        usage: &TokenUsage,
        options: &crate::PromptOptions,
    ) -> Result<(), anyhow::Error> {
        <Agent as TurnRuntime>::on_step_usage(self.agent, usage, options).await
    }
}

#[async_trait]
impl ToolExecutorPort<crate::PromptOptions> for AgentTurnPorts<'_> {
    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        options: &crate::PromptOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome, anyhow::Error> {
        <Agent as TurnRuntime>::run_tools(self.agent, tool_calls, options, cancel).await
    }
}

impl EventSinkPort<crate::PromptOptions> for AgentTurnPorts<'_> {
    fn on_step_begin(
        &self,
        turn_id: zene_turn::TurnId,
        step_id: zene_turn::StepId,
        step: u32,
        options: &crate::PromptOptions,
    ) {
        <Agent as TurnRuntime>::on_step_begin(self.agent, turn_id, step_id, step, options);
    }
}

#[async_trait]
impl TurnRuntime for Agent {
    type Options = crate::PromptOptions;

    fn max_steps(&self) -> u32 {
        self.config.max_turns
    }

    fn active_turn(&mut self) -> Option<&mut zene_turn::TurnState> {
        self.active_turn.as_mut()
    }

    fn on_step_begin(
        &self,
        turn_id: zene_turn::TurnId,
        step_id: zene_turn::StepId,
        step: u32,
        options: &Self::Options,
    ) {
        emit_event(
            &options.event_handler,
            AgentEvent::StepBegin {
                turn_id,
                step_id,
                step,
            },
        );
    }

    async fn prepare_turn(&mut self, user_input: &str) -> Result<(), anyhow::Error> {
        turn_session::prepare_turn(
            turn_session::PrepareTurnDeps {
                session: &mut self.session,
                system_prompt: &self.system_prompt,
                usage_accumulator: &mut self.usage_accumulator,
                tool_dedup: &mut self.tool_dedup,
                resume_existing_turn: &mut self.resume_existing_turn,
            },
            user_input,
        );
        Ok(())
    }

    async fn run_step(
        &mut self,
        options: &Self::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<StepResult, anyhow::Error> {
        self.record_step_started()?;
        let context = self.prepare_step_context(options, cancel).await?;
        self.invoke_model(context, options, cancel).await
    }

    async fn on_step_usage(
        &mut self,
        usage: &TokenUsage,
        options: &Self::Options,
    ) -> Result<(), anyhow::Error> {
        let plan_mode_active = self.is_plan_mode_active();
        turn_session::record_step_usage(
            turn_session::StepUsageDeps {
                config: &self.config,
                context: &mut self.context,
                session: &mut self.session,
                tools: self.tools.as_ref(),
                tool_policy: self.runtime_scope.tool_policy,
                plan_mode_active,
                usage_accumulator: &mut self.usage_accumulator,
            },
            usage,
            options,
        )
    }

    async fn run_tools(
        &mut self,
        tool_calls: &[ToolCall],
        options: &Self::Options,
        cancel: Option<&CancellationToken>,
    ) -> Result<ToolBatchOutcome, anyhow::Error> {
        Agent::run_tools(self, tool_calls, options, cancel).await
    }

    fn inject_steer(&mut self, options: &Self::Options) -> Result<bool, anyhow::Error> {
        self.inject_pending_steer(options)
    }

    fn push_assistant(&mut self, message: zene_llm::Message) {
        self.session.push_message(message);
    }

    fn on_incomplete_turn(
        &mut self,
        max_steps: u32,
        final_text: &mut String,
        options: &Self::Options,
    ) -> Result<(), anyhow::Error> {
        turn_session::on_incomplete_turn(&mut self.session, max_steps, final_text, options);
        Ok(())
    }

    async fn finish_turn(&mut self) -> Result<(), anyhow::Error> {
        self.sync_todos_to_session();
        self.save_session()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zene_turn::TurnEnginePorts;

    fn assert_turn_engine_ports<P: TurnEnginePorts>() {}

    #[test]
    fn agent_turn_ports_implements_turn_engine_ports() {
        assert_turn_engine_ports::<AgentTurnPorts<'static>>();
    }

    #[tokio::test]
    async fn prepare_context_projects_session_messages() {
        let workdir = tempfile::tempdir().expect("workdir");
        let config = zene_config::ZeneConfig {
            provider: "anthropic".into(),
            anthropic_api_key: Some("test-key".into()),
            ..Default::default()
        };
        let session = zene_session::SessionRecord::new(workdir.path());
        let mut agent = crate::AgentBuilder::new(
            config,
            zene_sandbox::LocalSandbox::new(workdir.path()),
            session,
            zene_permission::PermissionMode::BypassPermissions,
        )
        .without_mcp()
        .build()
        .await
        .expect("build agent");

        zene_turn::begin_turn(&mut agent.active_turn).expect("begin turn");
        <Agent as TurnRuntime>::prepare_turn(&mut agent, "hello from test")
            .await
            .expect("prepare turn");

        let mut ports = AgentTurnPorts::new(&mut agent);
        let context = ports
            .prepare_context(&crate::PromptOptions::default(), None)
            .await
            .expect("prepare context");

        assert!(
            context
                .messages
                .iter()
                .any(|message| message.content.as_deref() == Some("hello from test")),
            "assembler must project the user prompt instead of returning an empty context"
        );
        assert!(context.context_epoch.is_some());
        assert!(context.metadata.is_some());
    }
}
