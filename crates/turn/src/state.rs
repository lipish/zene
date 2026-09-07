use std::fmt;

use anyhow::{anyhow, Result};
use uuid::Uuid;

/// Stable identity for a persisted or running agent session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
    pub fn from_string(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable identity for a model tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolCallId(String);

impl Default for ToolCallId {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
    pub fn from_string(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifier for one user prompt → final assistant response cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TurnId(Uuid);

impl Default for TurnId {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifier for one LLM invocation within a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepId(Uuid);

impl Default for StepId {
    fn default() -> Self {
        Self::new()
    }
}

impl StepId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Tracks an in-flight turn (one active turn per agent).
#[derive(Debug)]
pub struct TurnState {
    pub turn_id: TurnId,
    pub step: u32,
    pub step_id: Option<StepId>,
}

impl TurnState {
    pub fn begin() -> Self {
        Self {
            turn_id: TurnId::new(),
            step: 0,
            step_id: None,
        }
    }

    pub fn next_step_id(&mut self) -> StepId {
        self.step += 1;
        let step_id = StepId::new();
        self.step_id = Some(step_id);
        step_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueueMode {
    #[default]
    OneAtATime,
    All,
}

/// Buffered steer messages injected between model steps.
#[derive(Debug, Default)]
pub struct SteerBuffer {
    pending: Vec<String>,
    mode: QueueMode,
}

impl SteerBuffer {
    pub fn push(&mut self, text: String) {
        self.pending.push(text);
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn take_all(&mut self) -> Vec<String> {
        let count = match self.mode {
            QueueMode::OneAtATime => 1,
            QueueMode::All => self.pending.len(),
        };
        self.pending.drain(..count).collect()
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }

    pub fn set_mode(&mut self, mode: QueueMode) {
        self.mode = mode;
    }
}

/// Follow-up messages injected only when a turn would otherwise settle.
#[derive(Debug, Default)]
pub struct FollowUpBuffer {
    pending: Vec<String>,
    mode: QueueMode,
}

impl FollowUpBuffer {
    pub fn push(&mut self, text: String) {
        self.pending.push(text);
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn take_all(&mut self) -> Vec<String> {
        let count = match self.mode {
            QueueMode::OneAtATime => 1,
            QueueMode::All => self.pending.len(),
        };
        self.pending.drain(..count).collect()
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }

    pub fn set_mode(&mut self, mode: QueueMode) {
        self.mode = mode;
    }
}

pub fn agent_busy_error() -> anyhow::Error {
    anyhow!(
        "agent busy: a turn is already in progress; use steer() or follow_up(), or wait for the turn to finish"
    )
}

pub fn steer_requires_active_turn() -> anyhow::Error {
    anyhow!("no turn in progress; use prompt() to start a new turn")
}

pub fn max_turns_notice(max_steps: u32) -> String {
    format!(
        "[notice] Reached max_turns ({max_steps}) without a final answer. Send a follow-up to continue, or raise max turns / use Unlimited."
    )
}

pub fn aborted_error() -> anyhow::Error {
    anyhow!("turn aborted")
}

pub fn begin_turn(active: &mut Option<TurnState>) -> Result<()> {
    if active.is_some() {
        return Err(agent_busy_error());
    }
    *active = Some(TurnState::begin());
    Ok(())
}

pub fn end_turn(active: &mut Option<TurnState>) {
    active.take();
}

pub fn is_cancelled(cancel: Option<&tokio_util::sync::CancellationToken>) -> bool {
    cancel.is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_second_active_turn() {
        let mut active = None;
        begin_turn(&mut active).expect("first turn");
        let err = begin_turn(&mut active).unwrap_err();
        assert!(err.to_string().contains("agent busy"));
        end_turn(&mut active);
        begin_turn(&mut active).expect("turn after end");
    }

    #[test]
    fn steer_buffer_fifo() {
        let mut buf = SteerBuffer::default();
        buf.push("first".into());
        buf.push("second".into());
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.take_all(), vec!["first"]);
        let drained = buf.take_all();
        assert_eq!(drained, vec!["second"]);
        assert!(buf.is_empty());
    }

    #[test]
    fn queue_modes_control_drain_size() {
        let mut steer = SteerBuffer::default();
        steer.push("first".into());
        steer.push("second".into());
        assert_eq!(steer.take_all(), vec!["first"]);
        steer.set_mode(QueueMode::All);
        assert_eq!(steer.take_all(), vec!["second"]);

        let mut follow_up = FollowUpBuffer::default();
        follow_up.push("first".into());
        follow_up.push("second".into());
        follow_up.set_mode(QueueMode::All);
        assert_eq!(follow_up.take_all(), vec!["first", "second"]);
    }
}
