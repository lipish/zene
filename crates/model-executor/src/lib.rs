use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use zene_llm::{
    ChatClient, ChatRequest, ChatResponse, ContextMetadata, Message, StreamEvent, TokenUsage,
    ToolCall, ToolDefinition,
};

/// Turn/runtime-facing model request (provider details stay behind adapters).
#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub stream: bool,
    pub context: Option<ContextMetadata>,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub message: Message,
    pub usage: Option<TokenUsage>,
}

impl From<ModelRequest> for ChatRequest {
    fn from(request: ModelRequest) -> Self {
        ChatRequest {
            model: request.model,
            messages: request.messages,
            tools: request.tools,
            stream: request.stream,
            context: request.context,
            reasoning_effort: request.reasoning_effort,
        }
    }
}

impl From<ChatRequest> for ModelRequest {
    fn from(request: ChatRequest) -> Self {
        ModelRequest {
            model: request.model,
            messages: request.messages,
            tools: request.tools,
            stream: request.stream,
            context: request.context,
            reasoning_effort: request.reasoning_effort,
        }
    }
}

impl From<ModelResponse> for ChatResponse {
    fn from(response: ModelResponse) -> Self {
        ChatResponse {
            message: response.message,
            usage: response.usage,
        }
    }
}

impl From<ChatResponse> for ModelResponse {
    fn from(response: ChatResponse) -> Self {
        ModelResponse {
            message: response.message,
            usage: response.usage,
        }
    }
}

/// Stream item type at the ModelExecutor boundary (provider `StreamEvent` today).
pub type ModelEvent = StreamEvent;
pub type ModelStream = Pin<Box<dyn Stream<Item = Result<ModelEvent>> + Send>>;

pub fn build_request(
    model: &str,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    stream: bool,
    context: Option<ContextMetadata>,
) -> ModelRequest {
    ModelRequest {
        model: model.to_string(),
        messages,
        tools,
        stream,
        context,
        reasoning_effort: None,
    }
}

pub fn build_request_with_reasoning(
    model: &str,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    stream: bool,
    context: Option<ContextMetadata>,
    reasoning_effort: Option<String>,
) -> ModelRequest {
    ModelRequest {
        model: model.to_string(),
        messages,
        tools,
        stream,
        context,
        reasoning_effort,
    }
}

#[derive(Debug, Default)]
pub struct OverflowRetryState {
    truncated: bool,
    summarized: bool,
}

impl OverflowRetryState {
    pub fn flags(&self) -> (bool, bool) {
        (self.truncated, self.summarized)
    }
    pub fn set_flags(&mut self, truncated: bool, summarized: bool) {
        self.truncated = truncated;
        self.summarized = summarized;
    }
}

#[async_trait]
pub trait ModelExecutor: Send + Sync {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse>;
    async fn stream(&self, request: ModelRequest) -> Result<ModelStream>;
}

pub struct ChatClientExecutor {
    client: Arc<ChatClient>,
}

impl ChatClientExecutor {
    pub fn new(client: Arc<ChatClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl ModelExecutor for ChatClientExecutor {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        Ok(self.client.chat(request.into()).await?.into())
    }
    async fn stream(&self, request: ModelRequest) -> Result<ModelStream> {
        self.client.chat_stream(request.into()).await
    }
}

#[derive(Default)]
pub struct StreamAccumulator {
    text: String,
    tool_calls: Vec<ToolCallBuilder>,
    usage: Option<zene_llm::TokenUsage>,
}

impl StreamAccumulator {
    pub fn apply(&mut self, event: &StreamEvent) -> bool {
        match event {
            StreamEvent::TextDelta(delta) => self.text.push_str(delta),
            StreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                while self.tool_calls.len() <= *index {
                    self.tool_calls.push(ToolCallBuilder::default());
                }
                apply_tool_call_delta(
                    &mut self.tool_calls[*index],
                    id.clone(),
                    name.clone(),
                    arguments.clone(),
                );
            }
            StreamEvent::Done { usage } => {
                self.usage = *usage;
                return true;
            }
            StreamEvent::ThoughtDelta(_) => {}
        }
        false
    }
    pub fn has_text(&self) -> bool {
        !self.text.is_empty()
    }
    pub fn finish(self) -> (Message, Option<zene_llm::TokenUsage>) {
        (assemble_message(self.text, self.tool_calls), self.usage)
    }
}

#[derive(Default)]
pub struct ToolCallBuilder {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub fn apply_tool_call_delta(
    call: &mut ToolCallBuilder,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
) {
    if let Some(id) = id {
        call.id = id;
    }
    if let Some(name) = name {
        call.name = name;
    }
    if let Some(arguments) = arguments {
        call.arguments.push_str(&arguments);
    }
}

pub fn assemble_message(text: String, builders: Vec<ToolCallBuilder>) -> Message {
    let calls = normalize_tool_calls(
        builders
            .into_iter()
            .filter(|call| !call.name.is_empty())
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
    );
    if calls.is_empty() {
        Message::assistant(text)
    } else {
        Message::assistant_with_tools((!text.is_empty()).then_some(text), calls)
    }
}

fn normalize_tool_calls(mut calls: Vec<ToolCall>) -> Vec<ToolCall> {
    let mut used_ids = HashSet::new();
    for (index, call) in calls.iter_mut().enumerate() {
        if call.id.trim().is_empty() {
            call.id = format!("call_{index}");
        }
        let base = call.id.clone();
        let mut unique = base.clone();
        let mut suffix = 0u32;
        while !used_ids.insert(unique.clone()) {
            suffix += 1;
            unique = format!("{base}_{suffix}");
        }
        call.id = unique;
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{stream, StreamExt};

    struct FakeExecutor;
    #[async_trait]
    impl ModelExecutor for FakeExecutor {
        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
            Ok(ModelResponse {
                message: Message::assistant(request.messages.len().to_string()),
                usage: None,
            })
        }
        async fn stream(&self, _request: ModelRequest) -> Result<ModelStream> {
            Ok(Box::pin(stream::iter([Ok(StreamEvent::Done {
                usage: None,
            })])))
        }
    }

    #[tokio::test]
    async fn fake_executor_covers_boundaries() {
        let request = build_request(
            "fake",
            vec![Message::user("hello")],
            Vec::new(),
            false,
            None,
        );
        let response = FakeExecutor.complete(request.clone()).await.unwrap();
        assert_eq!(response.message.content.as_deref(), Some("1"));
        let mut stream = FakeExecutor.stream(request).await.unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::Done { usage: None }
        ));
    }

    #[test]
    fn model_request_round_trips_chat_request() {
        let model = ModelRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")],
            tools: Vec::new(),
            stream: true,
            context: None,
            reasoning_effort: Some("high".into()),
        };
        let chat: ChatRequest = model.clone().into();
        let back: ModelRequest = chat.into();
        assert_eq!(back.model, "m");
        assert_eq!(back.messages.len(), 1);
        assert!(back.stream);
        assert_eq!(back.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn overflow_retry_state_round_trips_flags() {
        let mut state = OverflowRetryState::default();
        assert_eq!(state.flags(), (false, false));
        state.set_flags(true, false);
        assert_eq!(state.flags(), (true, false));
        state.set_flags(true, true);
        assert_eq!(state.flags(), (true, true));
    }

    #[test]
    fn assembles_streamed_tool_calls() {
        let mut accumulator = StreamAccumulator::default();
        accumulator.apply(&StreamEvent::TextDelta("hello ".into()));
        accumulator.apply(&StreamEvent::ToolCallDelta {
            index: 0,
            id: Some("call-1".into()),
            name: Some("Read".into()),
            arguments: Some("{}".into()),
        });
        assert!(accumulator.apply(&StreamEvent::Done { usage: None }));
        let (message, _) = accumulator.finish();
        assert_eq!(message.content.as_deref(), Some("hello "));
        assert_eq!(message.tool_calls.unwrap()[0].name, "Read");
    }
}
