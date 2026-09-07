//! Agent-specific actor that exclusively owns a [`zene_core::Agent`].

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use tokio::sync::{broadcast, oneshot, watch};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;

use zene_runtime::{
    ApprovalDecision, ApprovalWaiters, ExecutionState, RuntimeCommand, RuntimeCommandMessage,
    RuntimeCommandReceiver, RuntimeCommandRouter, RuntimeControl, RuntimeEventPublisher,
    RuntimeLifecycle, RuntimeRecoveryInfo, RuntimeResponse,
};
use zene_session::{AgentRecordWriter, ExecutionCheckpointState, RecoveryPlan};
use zene_turn::{RuntimeEvent, SessionId, SteerBuffer};

use zene_core::{Agent, PromptOptions, RecoveryDisposition, RecoverySnapshot};

async fn record_runtime_checkpoint(
    writer: &AgentRecordWriter,
    session_id: &str,
    state: ExecutionCheckpointState,
    detail: Option<&str>,
) -> Result<()> {
    use zene_session::RecordEntry;
    let suffix = detail.unwrap_or("shutdown");
    writer.append_execution_checkpoint(&RecordEntry::ExecutionCheckpoint {
        turn_id: session_id.to_string(),
        step_id: None,
        tool_call_id: None,
        state,
        idempotency_key: format!("runtime/{session_id}/{suffix}"),
        context_epoch: None,
        model_request_hash: None,
        ts: chrono::Utc::now(),
    })?;
    Ok(())
}

type RuntimeMessage = RuntimeCommandMessage;

/// Cloneable command/event/state handle for one long-lived runtime actor.
#[derive(Clone)]
pub struct RuntimeHandle {
    router: RuntimeCommandRouter,
    record_writer: AgentRecordWriter,
}

impl RuntimeHandle {
    /// Spawn an actor that exclusively owns `agent`.
    pub fn spawn(agent: Agent) -> (Self, JoinHandle<Result<()>>) {
        Self::spawn_internal(agent, None)
    }

    /// Spawn an actor and automatically resume one safe model-boundary turn.
    ///
    /// Only a single open turn with no active tool is eligible. The durable
    /// resume fence is claimed before the actor starts, so concurrent runtime
    /// instances cannot replay the same model request.
    pub fn spawn_with_automatic_recovery(agent: Agent) -> (Self, JoinHandle<Result<()>>) {
        let record_writer = agent.execution_record_writer();
        let candidate = record_writer
            .recovery_snapshot()
            .ok()
            .filter(|snapshot| {
                let plan = snapshot.plan();
                plan.automatic_resume_implemented
            })
            .and_then(|snapshot| snapshot.resume_candidates.into_iter().next())
            .filter(|candidate| record_writer.claim_safe_resume(candidate).unwrap_or(false));
        Self::spawn_internal(agent, candidate)
    }

    fn spawn_internal(
        mut agent: Agent,
        candidate: Option<zene_session::ResumeCandidate>,
    ) -> (Self, JoinHandle<Result<()>>) {
        if candidate.is_some() {
            agent.set_resume_existing_turn(true);
        }
        let record_writer = agent.execution_record_writer();
        let (router, command_rx, events, state_tx) = RuntimeCommandRouter::channel(32);
        let handle = Self {
            router: router.clone(),
            record_writer: record_writer.clone(),
        };
        let task = tokio::spawn(run_actor(
            agent,
            record_writer,
            command_rx,
            events,
            state_tx,
            candidate,
        ));
        (handle, task)
    }

    /// Send a transport-neutral command and await its acknowledgement/result.
    pub async fn command(&self, command: RuntimeCommand) -> Result<RuntimeResponse> {
        self.router.command(command).await
    }

    pub async fn prompt(&self, text: impl Into<String>) -> Result<String> {
        match self
            .command(RuntimeCommand::Prompt { text: text.into() })
            .await?
        {
            RuntimeResponse::Prompt { text } => Ok(text),
            _ => Err(anyhow!("runtime returned an invalid prompt response")),
        }
    }

    pub async fn steer(&self, text: impl Into<String>) -> Result<()> {
        self.command(RuntimeCommand::Steer { text: text.into() })
            .await
            .map(|_| ())
    }

    pub async fn follow_up(&self, text: impl Into<String>) -> Result<()> {
        self.command(RuntimeCommand::FollowUp { text: text.into() })
            .await
            .map(|_| ())
    }

    pub async fn set_steering_mode(&self, mode: zene_turn::QueueMode) -> Result<()> {
        self.command(RuntimeCommand::SetSteeringMode { mode })
            .await
            .map(|_| ())
    }

    pub async fn set_follow_up_mode(&self, mode: zene_turn::QueueMode) -> Result<()> {
        self.command(RuntimeCommand::SetFollowUpMode { mode })
            .await
            .map(|_| ())
    }

    pub async fn resume_safe_turn(&self) -> Result<String> {
        match self.command(RuntimeCommand::ResumeSafeTurn).await? {
            RuntimeResponse::Prompt { text } => Ok(text),
            _ => Err(anyhow!("runtime returned an invalid resume response")),
        }
    }

    pub async fn cancel(&self) -> Result<()> {
        self.command(RuntimeCommand::Cancel).await.map(|_| ())
    }

    pub async fn set_mode(&self, mode_id: impl Into<String>) -> Result<String> {
        match self
            .command(RuntimeCommand::SetMode {
                mode_id: mode_id.into(),
            })
            .await?
        {
            RuntimeResponse::Mode { mode_id } => Ok(mode_id),
            _ => Err(anyhow!("runtime returned an invalid mode response")),
        }
    }

    pub async fn current_mode(&self) -> Result<String> {
        match self.command(RuntimeCommand::GetMode).await? {
            RuntimeResponse::Mode { mode_id } => Ok(mode_id),
            _ => Err(anyhow!("runtime returned an invalid mode response")),
        }
    }

    pub async fn activate_tools(&self, names: Vec<String>) -> Result<Vec<String>> {
        match self
            .command(RuntimeCommand::ActivateTools { names })
            .await?
        {
            RuntimeResponse::Tools { names } => Ok(names),
            _ => Err(anyhow!(
                "runtime returned an invalid activate tools response"
            )),
        }
    }

    pub async fn deactivate_tools(&self, names: Vec<String>) -> Result<Vec<String>> {
        match self
            .command(RuntimeCommand::DeactivateTools { names })
            .await?
        {
            RuntimeResponse::Tools { names } => Ok(names),
            _ => Err(anyhow!(
                "runtime returned an invalid deactivate tools response"
            )),
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.command(RuntimeCommand::Shutdown).await.map(|_| ())
    }

    pub async fn approve(
        &self,
        request_id: impl Into<String>,
        decision: ApprovalDecision,
    ) -> Result<()> {
        self.command(RuntimeCommand::Approval {
            request_id: request_id.into(),
            decision,
        })
        .await
        .map(|_| ())
    }

    /// Subscribe to the ordered runtime event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<RuntimeEvent> {
        self.router.subscribe()
    }

    /// Subscribe to the latest execution state.
    pub fn state(&self) -> watch::Receiver<ExecutionState> {
        self.router.state()
    }

    /// Read the durable recovery view without starting or replaying execution.
    pub fn recovery_snapshot(&self) -> Result<RecoverySnapshot> {
        self.record_writer.recovery_snapshot()
    }

    /// Classify durable recovery state without starting or replaying execution.
    pub fn recovery_disposition(&self) -> Result<RecoveryDisposition> {
        Ok(self.recovery_snapshot()?.disposition())
    }

    /// Return the conservative recovery plan without starting execution.
    pub fn recovery_plan(&self) -> Result<RecoveryPlan> {
        Ok(self.recovery_snapshot()?.plan())
    }
}

fn terminal_lifecycle(cancelled: bool, result: &Result<String>) -> RuntimeLifecycle {
    match result {
        Ok(_) if !cancelled => RuntimeLifecycle::Completed,
        Ok(_) | Err(_) if cancelled => RuntimeLifecycle::Cancelled,
        Ok(_) => RuntimeLifecycle::Completed,
        Err(err) if err.to_string().contains("aborted") => RuntimeLifecycle::Cancelled,
        Err(err) => RuntimeLifecycle::Failed {
            message: err.to_string(),
        },
    }
}

fn recovery_disposition_name(disposition: zene_session::RecoveryDisposition) -> &'static str {
    match disposition {
        zene_session::RecoveryDisposition::Clean => "clean",
        zene_session::RecoveryDisposition::AlreadyCompleted => "already_completed",
        zene_session::RecoveryDisposition::SafeToResume => "safe_to_resume",
        zene_session::RecoveryDisposition::RequiresToolInspection => "requires_tool_inspection",
        zene_session::RecoveryDisposition::RequiresManualIntervention => {
            "requires_manual_intervention"
        }
    }
}

#[async_trait::async_trait]
impl RuntimeControl for RuntimeHandle {
    async fn prompt(&self, text: String) -> Result<String> {
        RuntimeHandle::prompt(self, text).await
    }

    async fn steer(&self, text: String) -> Result<()> {
        RuntimeHandle::steer(self, text).await
    }

    async fn follow_up(&self, text: String) -> Result<()> {
        RuntimeHandle::follow_up(self, text).await
    }

    async fn set_steering_mode(&self, mode: zene_turn::QueueMode) -> Result<()> {
        RuntimeHandle::set_steering_mode(self, mode).await
    }

    async fn set_follow_up_mode(&self, mode: zene_turn::QueueMode) -> Result<()> {
        RuntimeHandle::set_follow_up_mode(self, mode).await
    }

    async fn resume_safe_turn(&self) -> Result<String> {
        RuntimeHandle::resume_safe_turn(self).await
    }

    async fn cancel(&self) -> Result<()> {
        RuntimeHandle::cancel(self).await
    }

    async fn set_mode(&self, mode_id: String) -> Result<String> {
        RuntimeHandle::set_mode(self, mode_id).await
    }

    async fn activate_tools(&self, names: Vec<String>) -> Result<Vec<String>> {
        RuntimeHandle::activate_tools(self, names).await
    }

    async fn deactivate_tools(&self, names: Vec<String>) -> Result<Vec<String>> {
        RuntimeHandle::deactivate_tools(self, names).await
    }

    async fn current_mode(&self) -> Result<String> {
        RuntimeHandle::current_mode(self).await
    }

    async fn shutdown(&self) -> Result<()> {
        RuntimeHandle::shutdown(self).await
    }

    async fn approve(&self, request_id: String, decision: ApprovalDecision) -> Result<()> {
        RuntimeHandle::approve(self, request_id, decision).await
    }

    fn subscribe(&self) -> broadcast::Receiver<RuntimeEvent> {
        RuntimeHandle::subscribe(self)
    }

    fn recovery_info(&self) -> Result<RuntimeRecoveryInfo> {
        let snapshot = self.recovery_snapshot()?;
        let plan = snapshot.plan();
        Ok(RuntimeRecoveryInfo {
            disposition: recovery_disposition_name(plan.disposition).into(),
            has_incomplete_execution: snapshot.has_incomplete_execution(),
            active_turn_count: snapshot.active_turns.len(),
            active_tool_count: snapshot.active_tools.len(),
            safe_resume_allowed: plan.safe_resume_allowed,
            automatic_resume: plan.automatic_resume_implemented,
            reason: plan.reason,
        })
    }
}

struct PendingPrompt {
    text: String,
    reply: oneshot::Sender<std::result::Result<RuntimeResponse, String>>,
}

struct ActivePrompt {
    cancel: CancellationToken,
    reply: oneshot::Sender<std::result::Result<RuntimeResponse, String>>,
    task: JoinHandle<(Agent, Result<String>)>,
    waiters: Arc<ApprovalWaiters>,
}

enum ActivePoll {
    Finished(Box<std::result::Result<(Agent, Result<String>), JoinError>>),
    Command(Option<RuntimeMessage>),
}

async fn run_actor(
    agent: Agent,
    record_writer: AgentRecordWriter,
    mut commands: RuntimeCommandReceiver,
    events: broadcast::Sender<RuntimeEvent>,
    state: watch::Sender<ExecutionState>,
    initial_resume: Option<zene_session::ResumeCandidate>,
) -> Result<()> {
    let steer_buffer = agent.steer_buffer();
    let session_id = SessionId::from_string(agent.session().meta.id.clone());
    let publisher = RuntimeEventPublisher::new(events.clone(), state.clone(), session_id.clone());
    let mut agent = Some(agent);
    let mut queued: VecDeque<PendingPrompt> = VecDeque::new();
    let mut follow_up: VecDeque<PendingPrompt> = VecDeque::new();
    let mut active: Option<ActivePrompt> = initial_resume.map(|candidate| {
        let (reply, _response) = oneshot::channel();
        start_prompt(
            agent.take().expect("initial recovery owns agent"),
            PendingPrompt {
                text: candidate.prompt,
                reply,
            },
            &publisher,
        )
    });
    let mut shutdown_requested = false;

    loop {
        if active.is_none() {
            if shutdown_requested {
                while let Some(prompt) = queued.pop_front() {
                    let _ = prompt.reply.send(Err("runtime is shutting down".into()));
                }
                let _ = record_runtime_checkpoint(
                    &record_writer,
                    session_id.as_str(),
                    ExecutionCheckpointState::RuntimeShutdown,
                    Some("shutdown"),
                )
                .await;
                publisher.publish_lifecycle(RuntimeLifecycle::Shutdown);
                if let Some(mut agent) = agent.take() {
                    agent.shutdown().await?;
                }
                return Ok(());
            }

            if let Some(prompt) = queued.pop_front() {
                active = Some(start_prompt(
                    agent.take().expect("idle actor owns agent"),
                    prompt,
                    &publisher,
                ));
                continue;
            }

            let Some(message) = commands.recv().await else {
                shutdown_requested = true;
                continue;
            };
            active = handle_idle_command(
                &mut agent,
                message,
                &mut queued,
                &mut follow_up,
                &steer_buffer,
                &mut shutdown_requested,
                &publisher,
            );
            continue;
        }

        let poll = {
            let current = active.as_mut().expect("active prompt exists");
            tokio::select! {
                result = &mut current.task => ActivePoll::Finished(Box::new(result)),
                message = commands.recv() => ActivePoll::Command(message),
            }
        };

        match poll {
            ActivePoll::Finished(result) => {
                let result = *result;
                let current = active.take().expect("active prompt exists");
                let cancelled = current.cancel.is_cancelled();
                match result {
                    Ok((finished_agent, prompt_result)) => {
                        agent = Some(finished_agent);
                        let lifecycle = terminal_lifecycle(cancelled, &prompt_result);
                        let should_follow_up = matches!(&lifecycle, RuntimeLifecycle::Completed)
                            && !follow_up.is_empty();
                        let response = match (lifecycle, prompt_result) {
                            (RuntimeLifecycle::Completed, Ok(text)) => {
                                publisher.publish_lifecycle(RuntimeLifecycle::Completed);
                                Ok(RuntimeResponse::Prompt { text })
                            }
                            (RuntimeLifecycle::Cancelled, _) => {
                                publisher.publish_lifecycle(RuntimeLifecycle::Cancelled);
                                Err("turn cancelled".into())
                            }
                            (RuntimeLifecycle::Failed { message }, Err(_)) => {
                                publisher.publish_lifecycle(RuntimeLifecycle::Failed {
                                    message: message.clone(),
                                });
                                Err(message)
                            }
                            (RuntimeLifecycle::Completed, Err(_))
                            | (RuntimeLifecycle::Failed { .. }, Ok(_)) => {
                                unreachable!("terminal lifecycle must match prompt result")
                            }
                            (RuntimeLifecycle::Shutdown, _) => {
                                publisher.publish_lifecycle(RuntimeLifecycle::Shutdown);
                                Err("runtime shutdown".into())
                            }
                        };
                        let _ = current.reply.send(response);
                        if should_follow_up {
                            active = Some(start_prompt(
                                agent.take().expect("completed actor owns agent"),
                                follow_up.pop_front().expect("follow-up exists"),
                                &publisher,
                            ));
                        }
                    }
                    Err(err) => {
                        let message = format!("runtime turn task failed: {err}");
                        publisher.publish_lifecycle(RuntimeLifecycle::Failed {
                            message: message.clone(),
                        });
                        let _ = current.reply.send(Err(message));
                        // The task owned the Agent; after a panic it cannot be
                        // recovered. Stop cleanly instead of panicking again
                        // in the shutdown path.
                        let _ = record_runtime_checkpoint(
                            &record_writer,
                            session_id.as_str(),
                            ExecutionCheckpointState::RuntimeFailed,
                            Some("task_failed"),
                        )
                        .await;
                        publisher.publish_lifecycle(RuntimeLifecycle::Shutdown);
                        return Ok(());
                    }
                }
            }
            ActivePoll::Command(Some(message)) => {
                let current = active.as_ref().expect("active prompt exists");
                handle_active_command(
                    message,
                    &mut queued,
                    &mut follow_up,
                    &steer_buffer,
                    current.cancel.clone(),
                    &current.waiters,
                    &mut shutdown_requested,
                );
            }
            ActivePoll::Command(None) => {
                if let Some(current) = active.as_ref() {
                    current.cancel.cancel();
                }
                shutdown_requested = true;
            }
        }
    }
}

fn start_prompt(
    mut agent: Agent,
    prompt: PendingPrompt,
    publisher: &RuntimeEventPublisher,
) -> ActivePrompt {
    publisher.set_state(ExecutionState::Starting);
    let event_handler = publisher.handler();
    let waiters = Arc::new(ApprovalWaiters::new());
    if agent.runtime_approval_waiters() {
        agent.set_approval_broker(Arc::new(crate::approval::RuntimeOwnedBroker::new(
            Arc::clone(&waiters),
            publisher.clone(),
        )));
    }
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        let mut agent = agent;
        let result = agent
            .prompt(
                &prompt.text,
                PromptOptions {
                    stream: true,
                    cancel: Some(task_cancel),
                    event_handler: None,
                    runtime_event_handler: Some(event_handler),
                    quiet: true,
                },
            )
            .await;
        (agent, result)
    });
    ActivePrompt {
        cancel,
        reply: prompt.reply,
        task,
        waiters,
    }
}

fn handle_idle_command(
    agent: &mut Option<Agent>,
    message: RuntimeMessage,
    _queued: &mut VecDeque<PendingPrompt>,
    _follow_up: &mut VecDeque<PendingPrompt>,
    _steer_buffer: &std::sync::Arc<parking_lot::Mutex<SteerBuffer>>,
    shutdown_requested: &mut bool,
    publisher: &RuntimeEventPublisher,
) -> Option<ActivePrompt> {
    match message.command {
        RuntimeCommand::ResumeSafeTurn => {
            let snapshot = match agent
                .as_ref()
                .expect("idle actor owns agent")
                .execution_record_writer()
                .recovery_snapshot()
            {
                Ok(snapshot) => snapshot,
                Err(err) => {
                    let _ = message.reply.send(Err(err.to_string()));
                    return None;
                }
            };
            let plan = snapshot.plan();
            let Some(candidate) = snapshot
                .resume_candidates
                .into_iter()
                .next()
                .filter(|_| plan.automatic_resume_implemented)
            else {
                let _ = message.reply.send(Err(
                    "no unique prompt-backed safe resume candidate is available".into(),
                ));
                return None;
            };
            let writer = agent
                .as_ref()
                .expect("idle actor owns agent")
                .execution_record_writer();
            match writer.claim_safe_resume(&candidate) {
                Ok(true) => {}
                Ok(false) => {
                    let _ = message.reply.send(Err(
                        "safe resume rejected: candidate is already claimed or stale".into(),
                    ));
                    return None;
                }
                Err(err) => {
                    let _ = message
                        .reply
                        .send(Err(format!("safe resume fencing failed: {err}")));
                    return None;
                }
            }
            let mut resumed = agent.take().expect("idle actor owns agent");
            resumed.set_resume_existing_turn(true);
            Some(start_prompt(
                resumed,
                PendingPrompt {
                    text: candidate.prompt,
                    reply: message.reply,
                },
                publisher,
            ))
        }
        RuntimeCommand::Prompt { text } => {
            if text.trim().is_empty() {
                let _ = message.reply.send(Err("prompt cannot be empty".into()));
                None
            } else {
                Some(start_prompt(
                    agent.take().expect("idle actor owns agent"),
                    PendingPrompt {
                        text,
                        reply: message.reply,
                    },
                    publisher,
                ))
            }
        }
        RuntimeCommand::Steer { .. } => {
            let _ = message.reply.send(Err(
                "no turn in progress; use prompt() to start a new turn".into(),
            ));
            None
        }
        RuntimeCommand::FollowUp { .. } => {
            let _ = message.reply.send(Err(
                "no turn in progress; use prompt() to start a new turn".into(),
            ));
            None
        }
        RuntimeCommand::SetSteeringMode { mode } => {
            agent
                .as_ref()
                .expect("idle actor owns agent")
                .set_steering_mode(mode);
            let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
            None
        }
        RuntimeCommand::SetFollowUpMode { mode } => {
            agent
                .as_ref()
                .expect("idle actor owns agent")
                .set_follow_up_mode(mode);
            let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
            None
        }
        RuntimeCommand::Cancel => {
            let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
            None
        }
        RuntimeCommand::SetMode { mode_id } => match agent
            .as_mut()
            .expect("idle actor owns agent")
            .set_session_mode(&mode_id)
        {
            Ok(active) => {
                let _ = message.reply.send(Ok(RuntimeResponse::Mode {
                    mode_id: active.clone(),
                }));
                publish_state_event(publisher, &active);
                None
            }
            Err(err) => {
                let _ = message.reply.send(Err(err.to_string()));
                None
            }
        },
        RuntimeCommand::ActivateTools { names } => {
            let names = agent
                .as_ref()
                .expect("idle actor owns agent")
                .activate_tools(&names);
            let _ = message.reply.send(Ok(RuntimeResponse::Tools { names }));
            None
        }
        RuntimeCommand::DeactivateTools { names } => {
            let names = agent
                .as_ref()
                .expect("idle actor owns agent")
                .deactivate_tools(&names);
            let _ = message.reply.send(Ok(RuntimeResponse::Tools { names }));
            None
        }
        RuntimeCommand::Approval { request_id, .. } => {
            let _ = message
                .reply
                .send(Err(format!("no approval request {request_id} is pending")));
            None
        }
        RuntimeCommand::GetMode => {
            let mode_id = agent
                .as_ref()
                .expect("idle actor owns agent")
                .current_session_mode()
                .to_string();
            let _ = message.reply.send(Ok(RuntimeResponse::Mode { mode_id }));
            None
        }
        RuntimeCommand::Shutdown => {
            *shutdown_requested = true;
            let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
            None
        }
    }
}

fn handle_active_command(
    message: RuntimeMessage,
    queued: &mut VecDeque<PendingPrompt>,
    follow_up: &mut VecDeque<PendingPrompt>,
    steer_buffer: &std::sync::Arc<parking_lot::Mutex<SteerBuffer>>,
    cancel: CancellationToken,
    waiters: &ApprovalWaiters,
    shutdown_requested: &mut bool,
) {
    match message.command {
        RuntimeCommand::Prompt { text } => {
            if text.trim().is_empty() {
                let _ = message.reply.send(Err("prompt cannot be empty".into()));
            } else {
                queued.push_back(PendingPrompt {
                    text,
                    reply: message.reply,
                });
            }
        }
        RuntimeCommand::ResumeSafeTurn => {
            let _ = message
                .reply
                .send(Err("cannot resume while a turn is active".into()));
        }
        RuntimeCommand::Steer { text } => {
            let text = text.trim();
            if text.is_empty() {
                let _ = message
                    .reply
                    .send(Err("steer message cannot be empty".into()));
            } else {
                steer_buffer.lock().push(text.to_string());
                let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
            }
        }
        RuntimeCommand::FollowUp { text } => {
            let text = text.trim();
            if text.is_empty() {
                let _ = message
                    .reply
                    .send(Err("follow-up message cannot be empty".into()));
            } else {
                follow_up.push_back(PendingPrompt {
                    text: text.to_string(),
                    reply: message.reply,
                });
            }
        }
        RuntimeCommand::SetSteeringMode { .. } | RuntimeCommand::SetFollowUpMode { .. } => {
            let _ = message
                .reply
                .send(Err("cannot change queue mode while a turn is active".into()));
        }
        RuntimeCommand::Cancel => {
            waiters.cancel_all();
            cancel.cancel();
            queued.drain(..).for_each(|prompt| {
                let _ = prompt.reply.send(Err("turn cancelled".into()));
            });
            follow_up.drain(..).for_each(|prompt| {
                let _ = prompt.reply.send(Err("turn cancelled".into()));
            });
            steer_buffer.lock().clear();
            let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
        }
        RuntimeCommand::SetMode { .. }
        | RuntimeCommand::ActivateTools { .. }
        | RuntimeCommand::DeactivateTools { .. }
        | RuntimeCommand::GetMode => {
            let _ = message.reply.send(Err(
                "cannot change or read mode while a turn is active".into()
            ));
        }
        RuntimeCommand::Approval {
            request_id,
            decision,
        } => match waiters.resolve(&request_id, decision) {
            Ok(()) => {
                let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
            }
            Err(err) => {
                let _ = message.reply.send(Err(err.to_string()));
            }
        },
        RuntimeCommand::Shutdown => {
            *shutdown_requested = true;
            waiters.cancel_all();
            cancel.cancel();
            let _ = message.reply.send(Ok(RuntimeResponse::Accepted));
        }
    }
}

fn publish_state_event(publisher: &RuntimeEventPublisher, state: &str) {
    publisher.publish_state(state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use zene_permission::PromptChoice;
    use zene_turn::TurnId;

    #[test]
    fn recovery_disposition_names_are_protocol_stable() {
        let cases = [
            (zene_session::RecoveryDisposition::Clean, "clean"),
            (
                zene_session::RecoveryDisposition::AlreadyCompleted,
                "already_completed",
            ),
            (
                zene_session::RecoveryDisposition::SafeToResume,
                "safe_to_resume",
            ),
            (
                zene_session::RecoveryDisposition::RequiresToolInspection,
                "requires_tool_inspection",
            ),
            (
                zene_session::RecoveryDisposition::RequiresManualIntervention,
                "requires_manual_intervention",
            ),
        ];
        for (disposition, expected) in cases {
            assert_eq!(recovery_disposition_name(disposition), expected);
        }
    }

    #[test]
    fn approval_decisions_map_to_permission_choices() {
        assert_eq!(
            crate::approval::prompt_choice(ApprovalDecision::AllowOnce),
            PromptChoice::AllowOnce
        );
        assert_eq!(
            crate::approval::prompt_choice(ApprovalDecision::AllowSession),
            PromptChoice::AllowSession
        );
        assert_eq!(
            crate::approval::prompt_choice(ApprovalDecision::Deny),
            PromptChoice::Deny
        );
    }

    #[test]
    fn execution_state_separates_runtime_from_cloud_status() {
        let turn_id = TurnId::new();
        let state = ExecutionState::Running { turn_id, step: 2 };
        assert!(matches!(state, ExecutionState::Running { step: 2, .. }));
    }

    #[test]
    fn terminal_lifecycle_classifies_prompt_outcomes() {
        assert_eq!(
            terminal_lifecycle(false, &Ok("done".into())),
            RuntimeLifecycle::Completed
        );
        assert_eq!(
            terminal_lifecycle(true, &Ok("done".into())),
            RuntimeLifecycle::Cancelled
        );
        assert_eq!(
            terminal_lifecycle(false, &Err(anyhow!("aborted by caller"))),
            RuntimeLifecycle::Cancelled
        );
        assert_eq!(
            terminal_lifecycle(false, &Err(anyhow!("provider failed"))),
            RuntimeLifecycle::Failed {
                message: "provider failed".into(),
            }
        );
    }

    #[tokio::test]
    async fn runtime_handle_reads_recovery_without_starting_execution() {
        use chrono::Utc;
        use tempfile::tempdir;
        use zene_config::ZeneConfig;
        use zene_session::{
            AgentRecordWriter, ExecutionCheckpointState, RecordEntry, SessionRecord,
        };

        let workdir = tempdir().expect("workdir");
        let record_dir = tempdir().expect("record dir");
        let session = SessionRecord::new(workdir.path());
        let writer = AgentRecordWriter::from_path(record_dir.path().join("record.jsonl"))
            .expect("record writer");
        writer
            .append(&RecordEntry::ExecutionCheckpoint {
                turn_id: "turn-incomplete".into(),
                step_id: None,
                tool_call_id: None,
                state: ExecutionCheckpointState::TurnStarted,
                idempotency_key: "turn-incomplete/started".into(),
                context_epoch: None,
                model_request_hash: None,
                ts: Utc::now(),
            })
            .expect("checkpoint");

        let config = ZeneConfig {
            provider: "anthropic".into(),
            anthropic_api_key: Some("test-key".into()),
            ..Default::default()
        };
        let agent = zene_core::Agent::builder(workdir.path())
            .config(config)
            .session(session)
            .bypass_permissions()
            .without_mcp()
            .record_writer(writer)
            .build()
            .await
            .expect("build agent without network calls");
        let (runtime, task) = RuntimeHandle::spawn_with_automatic_recovery(agent);

        assert_eq!(
            runtime
                .recovery_disposition()
                .expect("recovery disposition"),
            RecoveryDisposition::SafeToResume
        );
        assert!(runtime
            .recovery_snapshot()
            .expect("recovery snapshot")
            .has_incomplete_execution());
        runtime.shutdown().await.expect("shutdown");
        task.await.expect("actor join").expect("actor result");
    }

    #[tokio::test]
    async fn runtime_handle_reports_unfinished_resume_claim_for_recovery() {
        use chrono::Utc;
        use tempfile::tempdir;
        use zene_config::ZeneConfig;
        use zene_session::{
            AgentRecordWriter, ExecutionCheckpointState, RecordEntry, SessionRecord,
        };

        let workdir = tempdir().expect("workdir");
        let record_dir = tempdir().expect("record dir");
        let session = SessionRecord::new(workdir.path());
        let writer = AgentRecordWriter::from_path(record_dir.path().join("record.jsonl"))
            .expect("record writer");
        writer
            .append(&RecordEntry::ExecutionCheckpoint {
                turn_id: "turn-claimed".into(),
                step_id: None,
                tool_call_id: None,
                state: ExecutionCheckpointState::TurnResumed,
                idempotency_key: "resume/turn-claimed".into(),
                context_epoch: None,
                model_request_hash: None,
                ts: Utc::now(),
            })
            .expect("resume claim");

        let config = ZeneConfig {
            provider: "anthropic".into(),
            anthropic_api_key: Some("test-key".into()),
            ..Default::default()
        };
        let agent = zene_core::Agent::builder(workdir.path())
            .config(config)
            .session(session)
            .bypass_permissions()
            .without_mcp()
            .record_writer(writer)
            .build()
            .await
            .expect("build agent without network calls");
        let (runtime, task) = RuntimeHandle::spawn(agent);

        assert_eq!(
            runtime
                .recovery_disposition()
                .expect("recovery disposition"),
            RecoveryDisposition::RequiresManualIntervention
        );
        assert!(runtime
            .recovery_snapshot()
            .expect("recovery snapshot")
            .has_incomplete_execution());
        runtime.shutdown().await.expect("shutdown");
        task.await.expect("actor join").expect("actor result");
    }

    #[tokio::test]
    async fn active_cancel_acknowledges_and_cancels_prompt_token() {
        let (reply, response) = oneshot::channel();
        let cancel = CancellationToken::new();
        let cancel_for_assertion = cancel.clone();
        let mut queued = VecDeque::new();
        let steer = Arc::new(parking_lot::Mutex::new(SteerBuffer::default()));
        let waiters = ApprovalWaiters::new();
        let pending = waiters.register("req-1".into());

        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Cancel,
                reply,
            },
            &mut queued,
            &mut VecDeque::new(),
            &steer,
            cancel,
            &waiters,
            &mut false,
        );

        assert!(cancel_for_assertion.is_cancelled());
        assert!(matches!(
            response.await.unwrap(),
            Ok(RuntimeResponse::Accepted)
        ));
        assert!(queued.is_empty());
        assert!(pending.await.is_err());
        assert!(waiters.is_empty());
    }

    #[tokio::test]
    async fn active_mode_commands_are_rejected_without_mutating_queue() {
        for command in [
            RuntimeCommand::SetMode {
                mode_id: "plan".into(),
            },
            RuntimeCommand::GetMode,
        ] {
            let (reply, response) = oneshot::channel();
            let cancel = CancellationToken::new();
            let mut queued = VecDeque::new();
            let steer = Arc::new(parking_lot::Mutex::new(SteerBuffer::default()));

            handle_active_command(
                RuntimeMessage { command, reply },
                &mut queued,
                &mut VecDeque::new(),
                &steer,
                cancel,
                &ApprovalWaiters::new(),
                &mut false,
            );

            let error = response.await.unwrap().unwrap_err();
            assert!(error.contains("active"));
            assert!(queued.is_empty());
        }
    }

    #[tokio::test]
    async fn active_steer_is_buffered_and_empty_steer_is_rejected() {
        let steer = Arc::new(parking_lot::Mutex::new(SteerBuffer::default()));
        let cancel = CancellationToken::new();
        let mut queued = VecDeque::new();
        let (reply, response) = oneshot::channel();
        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Steer {
                    text: "  keep going  ".into(),
                },
                reply,
            },
            &mut queued,
            &mut VecDeque::new(),
            &steer,
            cancel.clone(),
            &ApprovalWaiters::new(),
            &mut false,
        );
        assert!(matches!(
            response.await.unwrap(),
            Ok(RuntimeResponse::Accepted)
        ));
        assert_eq!(steer.lock().take_all(), vec!["keep going"]);

        let (reply, response) = oneshot::channel();
        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Steer { text: "  ".into() },
                reply,
            },
            &mut queued,
            &mut VecDeque::new(),
            &steer,
            cancel,
            &ApprovalWaiters::new(),
            &mut false,
        );
        assert!(response.await.unwrap().unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn active_prompt_queue_rejects_empty_and_preserves_valid_prompt() {
        let steer = Arc::new(parking_lot::Mutex::new(SteerBuffer::default()));
        let cancel = CancellationToken::new();
        let mut queued = VecDeque::new();

        let (reply, response) = oneshot::channel();
        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Prompt { text: "  ".into() },
                reply,
            },
            &mut queued,
            &mut VecDeque::new(),
            &steer,
            cancel.clone(),
            &ApprovalWaiters::new(),
            &mut false,
        );
        assert!(response.await.unwrap().unwrap_err().contains("empty"));
        assert!(queued.is_empty());

        let (reply, _response) = oneshot::channel();
        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Prompt {
                    text: "next".into(),
                },
                reply,
            },
            &mut queued,
            &mut VecDeque::new(),
            &steer,
            cancel,
            &ApprovalWaiters::new(),
            &mut false,
        );
        assert_eq!(queued.len(), 1);
        assert_eq!(queued.front().unwrap().text, "next");
    }

    #[tokio::test]
    async fn active_approval_resolves_registered_waiter() {
        let waiters = ApprovalWaiters::new();
        let rx = waiters.register("req-1".into());
        let (reply, response) = oneshot::channel();
        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Approval {
                    request_id: "req-1".into(),
                    decision: ApprovalDecision::AllowOnce,
                },
                reply,
            },
            &mut VecDeque::new(),
            &mut VecDeque::new(),
            &Arc::new(parking_lot::Mutex::new(SteerBuffer::default())),
            CancellationToken::new(),
            &waiters,
            &mut false,
        );
        assert!(matches!(
            response.await.unwrap(),
            Ok(RuntimeResponse::Accepted)
        ));
        assert_eq!(rx.await.unwrap(), ApprovalDecision::AllowOnce);
        assert!(waiters.is_empty());
    }

    #[tokio::test]
    async fn active_approval_unknown_request_is_rejected() {
        let waiters = ApprovalWaiters::new();
        let (reply, response) = oneshot::channel();
        handle_active_command(
            RuntimeMessage {
                command: RuntimeCommand::Approval {
                    request_id: "missing".into(),
                    decision: ApprovalDecision::Deny,
                },
                reply,
            },
            &mut VecDeque::new(),
            &mut VecDeque::new(),
            &Arc::new(parking_lot::Mutex::new(SteerBuffer::default())),
            CancellationToken::new(),
            &waiters,
            &mut false,
        );
        assert!(response
            .await
            .unwrap()
            .unwrap_err()
            .contains("no approval request missing is pending"));
    }
}
