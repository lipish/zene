use std::pin::Pin;

use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use zene_config::{ProviderKind, ZeneConfig};

use crate::anthropic::AnthropicProvider;
use crate::openai_compatible::OpenAiCompatibleProvider;
use crate::provider::{ChatRequest, ChatResponse, Provider, StreamEvent};

/// Unified LLM client that delegates to the configured provider.
pub struct ChatClient {
    inner: Box<dyn Provider>,
}

impl ChatClient {
    pub async fn from_config(config: &ZeneConfig) -> Result<Self> {
        let inner: Box<dyn Provider> = match config.provider_kind() {
            ProviderKind::OpenAi => Box::new(OpenAiCompatibleProvider::from_config(config).await?),
            ProviderKind::Anthropic => Box::new(AnthropicProvider::from_config(config)?),
        };
        Ok(Self { inner })
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.inner.chat(request).await
    }

    pub async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        self.inner.chat_stream(request).await
    }
}

/// Select provider implementation from config (for tests).
pub async fn provider_from_config(config: &ZeneConfig) -> Result<Box<dyn Provider>> {
    match config.provider_kind() {
        ProviderKind::OpenAi => Ok(Box::new(
            OpenAiCompatibleProvider::from_config(config).await?,
        )),
        ProviderKind::Anthropic => Ok(Box::new(AnthropicProvider::from_config(config)?)),
    }
}

/// Returns the provider kind that would be selected for the given config.
pub fn selected_provider_kind(config: &ZeneConfig) -> ProviderKind {
    config.provider_kind()
}

#[async_trait]
impl Provider for ChatClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.inner.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        self.inner.chat_stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zene_config::ProviderKind;

    #[test]
    fn provider_selection_openai_default() {
        let config = ZeneConfig::default();
        assert_eq!(selected_provider_kind(&config), ProviderKind::OpenAi);
    }

    #[test]
    fn provider_selection_anthropic_from_config() {
        let config = ZeneConfig {
            provider: "anthropic".to_string(),
            ..Default::default()
        };
        assert_eq!(selected_provider_kind(&config), ProviderKind::Anthropic);
    }

    #[test]
    fn provider_selection_openai_compatible_alias() {
        let config = ZeneConfig {
            provider: "openai-compatible".to_string(),
            ..Default::default()
        };
        assert_eq!(selected_provider_kind(&config), ProviderKind::OpenAi);
    }

    #[test]
    fn unknown_provider_errors() {
        let config = ZeneConfig {
            provider: "unknown-vendor".to_string(),
            ..Default::default()
        };
        let err = config.provider_kind_parse().unwrap_err();
        assert!(err.to_string().contains("unknown provider"));
    }
}
