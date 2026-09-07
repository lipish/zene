//! Context orchestration: estimate, compact, memory, prefire, epoch.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use zene_llm::{ContextMetadata, Message, TokenUsage, ToolDefinition};

use crate::assemble::{
    assemble_outbound, delivery_mode_from_env, stable_system_boundary, DeliveryMode,
};
use crate::compaction::{
    apply_steps_truncate_pass, compact_session, compact_session_forced, is_context_overflow_error,
    CompactionOptions, CompactionParams, CompactionResult,
};
use crate::config::CompactionConfig;
use crate::context_water::ContextWaterLevel;
use crate::event_handler::{ContextEventHandler, EventOutcome};
use crate::events::ContextEvent;
#[cfg(feature = "gateway")]
use crate::gateway::gateway_configured;
#[cfg(not(feature = "gateway"))]
use crate::gateway_stub::gateway_configured;
use crate::hooks::ContextHooks;
use crate::layout::{
    apply_tail_decorations, classify_prefix_break, content_is_reminder,
    prefix_adjacent_decoration_index, prefix_fingerprint, relocate_prefix_adjacent_decorations,
    split_layout, PrefixCacheExplain,
};
#[cfg(feature = "memory")]
use crate::memory;
#[cfg(not(feature = "memory"))]
use crate::memory_stub as memory;
use crate::model::ContextModel;
#[cfg(feature = "prefire")]
use crate::prefire::{self, PrefireCache, PrefireState};
#[cfg(not(feature = "prefire"))]
use crate::prefire_stub::{PrefireCache, PrefireState};
use crate::session::ContextSession;
use crate::tokens::{self, TokenEstimator};
#[cfg(feature = "prefire")]
use crate::two_pass;

fn projection_injected_labels(messages: &[Message]) -> Vec<String> {
    let mut labels = Vec::new();
    if messages
        .iter()
        .any(|message| message.kind == Some(zene_llm::MessageKind::CompactionSummary))
    {
        labels.push("compaction_summary".to_string());
    }
    if messages
        .iter()
        .any(|message| message.content.as_deref().is_some_and(content_is_reminder))
    {
        labels.push("system_reminder".to_string());
    }
    labels
}

fn projection_truncated_message_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|message| {
            message.content.as_deref().is_some_and(|content| {
                content.contains("[truncated ") || content.contains("…[steps-truncated ")
            })
        })
        .count()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutputProvenance {
    pub message_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Stable classification: `truncated` or `handle`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectedSource {
    pub message_index: usize,
    pub kind: String,
    pub source: String,
}

fn tool_handle_reference(content: &str) -> Option<String> {
    let marker = "[zene-tool-output path=\"";
    let start = content.find(marker)? + marker.len();
    let end = content[start..].find('\"')?;
    Some(content[start..start + end].to_string())
}

fn saved_output_reference(content: &str) -> Option<String> {
    let marker = "full output saved to ";
    let start = content.find(marker)? + marker.len();
    let reference = content[start..]
        .split_whitespace()
        .next()?
        .trim_end_matches(['.', ',', ';', ']', ')']);
    (!reference.is_empty()).then(|| reference.to_string())
}

fn projection_tool_output_provenance(messages: &[Message]) -> Vec<ToolOutputProvenance> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(message_index, message)| {
            if message.role != zene_llm::Role::Tool {
                return None;
            }
            let content = message.content.as_deref()?;
            let handle_reference =
                tool_handle_reference(content).or_else(|| saved_output_reference(content));
            let kind = if content.contains("[zene-tool-output ") {
                "handle"
            } else if content.contains("[truncated ") || content.contains("…[steps-truncated ") {
                "truncated"
            } else {
                return None;
            };
            Some(ToolOutputProvenance {
                message_index,
                tool_call_id: message.tool_call_id.clone(),
                tool_name: message.name.clone(),
                kind: kind.to_string(),
                handle_reference,
            })
        })
        .collect()
}

fn injected_source_kind(content: &str) -> Vec<&'static str> {
    let lower = content.to_ascii_lowercase();
    let mut sources = Vec::new();
    if lower.contains("<memory-context>") {
        sources.push("memory");
    }
    if lower.contains("todo") {
        sources.push("todos");
    }
    if lower.contains("background") || lower.contains("task") {
        sources.push("background_tasks");
    }
    if lower.contains("plan mode") || lower.contains("exitplanmode") {
        sources.push("plan");
    }
    if sources.is_empty() {
        sources.push("system");
    }
    sources
}

fn projection_injected_sources(messages: &[Message]) -> Vec<InjectedSource> {
    messages
        .iter()
        .enumerate()
        .flat_map(|(message_index, message)| {
            if message.kind == Some(zene_llm::MessageKind::CompactionSummary) {
                return vec![InjectedSource {
                    message_index,
                    kind: "compaction_summary".to_string(),
                    source: "compaction_event".to_string(),
                }];
            }
            let Some(content) = message.content.as_deref() else {
                return Vec::new();
            };
            if !content_is_reminder(content) {
                return Vec::new();
            }
            injected_source_kind(content)
                .into_iter()
                .map(|source| InjectedSource {
                    message_index,
                    kind: "system_reminder".to_string(),
                    source: source.to_string(),
                })
                .collect()
        })
        .collect()
}

/// Builds a dedicated client for prefire pass1 (runtime-provided; avoids `ZeneConfig` in this crate).
pub type PrefireClientFactory = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<Arc<dyn ContextModel>>> + Send>> + Send + Sync,
>;

/// Outbound view for one LLM step after context preparation.
#[derive(Debug, Clone)]
pub struct StepContext {
    pub messages: Vec<Message>,
    pub metadata: ContextMetadata,
    pub estimate_tokens: u32,
}

/// Read-only decision before context mutations are committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsageUpdate {
    pub context_tokens: u32,
    pub context_window: u32,
    pub context_percent: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextObservation {
    pub estimated_tokens: u32,
    pub should_compact: bool,
    pub preflight_overflow: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectionExplain {
    pub source_message_count: usize,
    pub projected_message_count: usize,
    pub source_event_count: usize,
    pub active_event_count: usize,
    pub cache_drift_detected: bool,
    pub used_materialized_fallback: bool,
    pub fallback_reason: Option<String>,
    pub active_branch_id: Option<String>,
    pub active_path_start_sequence: Option<u64>,
    /// Stable labels for projection decorations visible in the outbound messages.
    /// Current labels include `compaction_summary` and `system_reminder`.
    pub injected: Vec<String>,
    /// Number of messages retained in the outbound projection.
    pub retained_message_count: usize,
    /// Number of distinct turn IDs represented by the active event path.
    pub retained_turn_count: usize,
    /// Events excluded by branch/rewind projection boundaries.
    pub dropped_event_count: usize,
    /// Messages whose bodies were reduced by a truncation pass.
    pub truncated_message_count: usize,
    /// Compaction event IDs contributing to the active projection.
    pub compaction_event_ids: Vec<String>,
    /// Per-message provenance for bounded tool output.
    pub tool_output_provenance: Vec<ToolOutputProvenance>,
    /// Turn IDs represented by the active event path, in path order.
    pub retained_turn_ids: Vec<String>,
    /// Classification of injected projection decorations and their sources.
    pub injected_sources: Vec<InjectedSource>,
    pub delivery: DeliveryMode,
    pub delivery_tail_start: Option<usize>,
    pub estimate_tokens: u32,
    pub context_epoch: u64,
    /// Prefix-cache layout and predicted break versus the previous outbound call.
    pub prefix_cache: PrefixCacheExplain,
}

/// Result of [`ContextEngine::prepare_step`].
#[derive(Debug, Clone)]
pub struct PrepareStepResult {
    pub step: StepContext,
    pub compaction: Option<CompactionResult>,
    pub events: Vec<ContextEvent>,
    pub explain: ProjectionExplain,
}

#[derive(Debug, Clone)]
struct CommitResult {
    compaction: Option<CompactionResult>,
    events: Vec<ContextEvent>,
}

/// Result of [`ContextEngine::compact_forced`].
#[derive(Debug, Clone)]
pub struct ForcedCompactResult {
    pub compaction: Option<CompactionResult>,
    pub events: Vec<ContextEvent>,
}

/// Result of [`ContextEngine::handle_overflow`].
#[derive(Debug, Clone)]
pub struct OverflowHandleResult {
    pub retry: bool,
    pub compaction: Option<CompactionResult>,
    pub events: Vec<ContextEvent>,
}

/// Dependencies passed into context operations (session + runtime config).
pub struct ContextDeps<'a> {
    pub session: &'a mut dyn ContextSession,
    pub compaction_config: &'a CompactionConfig,
    pub model: &'a str,
    pub client: &'a dyn crate::model::ContextModel,
    pub hooks: Option<&'a dyn ContextHooks>,
    pub system_prompt: &'a str,
    pub estimator: &'a TokenEstimator,
    pub handler: &'a mut dyn ContextEventHandler,
    #[cfg(feature = "prefire")]
    pub prefire_client_factory: Option<PrefireClientFactory>,
}

/// Semantic context authority for one agent session.
pub struct ContextEngine {
    water: ContextWaterLevel,
    prefire: PrefireState,
    epoch: u64,
    last_memory_flush_compaction: u64,
    external_session_id: Option<String>,
    pending_publish: bool,
    gateway_prefix_len: usize,
    initial_publish_done: bool,
    tail_sections: Vec<String>,
    last_prefix_fingerprint: Option<String>,
    last_prefix_tokens: u32,
    last_epoch_bump_reason: Option<&'static str>,
    last_cached_tokens: Option<u64>,
    last_gateway_hit_tokens: Option<u64>,
    last_anchor_aligned: Option<bool>,
    last_unchanged_reprocessed_est: Option<u64>,
}

impl ContextEngine {
    pub fn new(context_window_tokens: u32) -> Self {
        Self {
            water: ContextWaterLevel::new(context_window_tokens),
            prefire: PrefireState::new(),
            epoch: 0,
            last_memory_flush_compaction: 0,
            external_session_id: None,
            pending_publish: false,
            gateway_prefix_len: 0,
            initial_publish_done: false,
            tail_sections: Vec::new(),
            last_prefix_fingerprint: None,
            last_prefix_tokens: 0,
            last_epoch_bump_reason: None,
            last_cached_tokens: None,
            last_gateway_hit_tokens: None,
            last_anchor_aligned: None,
            last_unchanged_reprocessed_est: None,
        }
    }

    /// Override session id for inference linkage (e.g. Cloud run_id).
    pub fn set_external_session_id(&mut self, id: Option<String>) {
        self.external_session_id = id;
    }

    pub fn water(&self) -> &ContextWaterLevel {
        &self.water
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn gateway_prefix_len(&self) -> usize {
        self.gateway_prefix_len
    }

    pub fn set_window(&mut self, context_window_tokens: u32) {
        self.water.set_window(context_window_tokens);
    }

    pub fn clear_prefire(&self) {
        self.prefire.clear();
    }

    pub fn prefire_has_cache(&self) -> bool {
        self.prefire.has_cache()
    }

    pub fn prefire_in_flight(&self) -> bool {
        self.prefire.is_in_flight()
    }

    pub fn restore_water_from_session(&mut self, tokens: u32) {
        self.water.last_prompt_tokens = Some(tokens);
        self.water.last_estimate_tokens = Some(tokens);
    }

    pub fn metadata(&self, session: &dyn ContextSession) -> ContextMetadata {
        let session_id = self.session_id_for(session);
        ContextMetadata::new(session_id, self.epoch)
    }

    fn metadata_for_outbound(
        &self,
        session: &dyn ContextSession,
        assembled: &crate::assemble::AssembledOutbound,
    ) -> ContextMetadata {
        let mut meta = self.metadata(session);
        meta.prefix_hash = assembled.prefix_hash.clone();
        meta.delivery = match assembled.mode {
            DeliveryMode::Full => zene_llm::ContextDelivery::Full,
            DeliveryMode::Delta => zene_llm::ContextDelivery::Delta,
        };
        meta.tail_start = assembled.tail_start;
        meta
    }

    pub fn on_system_prefix_changed(&mut self, reason: &'static str) -> ContextEvent {
        let old = self.epoch;
        self.epoch = self.epoch.saturating_add(1);
        self.pending_publish = true;
        self.last_epoch_bump_reason = Some(reason);
        ContextEvent::EpochBumped {
            old,
            new: self.epoch,
            reason,
        }
    }

    /// Live tail decorations for the next `project` / `assemble_step` call.
    /// `prepare_step` overwrites this from [`ContextHooks::step_tail_decorations`].
    pub fn set_step_tail_decorations(&mut self, sections: Vec<String>) {
        self.tail_sections = sections;
    }

    /// Re-assemble outbound view after session mutation (e.g. overflow compact).
    pub fn assemble_step(
        &mut self,
        session: &dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
    ) -> StepContext {
        let step = self.project(session, tools, estimator);
        let explain = self.explain_projection_for_step(session, &step);
        self.remember_projection(&step, &explain);
        step
    }

    /// Strict event-backed variant used by new model requests.
    pub fn try_assemble_step(
        &mut self,
        session: &dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
    ) -> Result<StepContext> {
        let _ = session.try_view().map_err(|reason| {
            anyhow!(
                "event-backed context projection unavailable: {}",
                reason.as_str()
            )
        })?;
        Ok(self.assemble_step(session, tools, estimator))
    }

    /// Observe session pressure without mutating the session.
    pub fn observe(
        &mut self,
        session: &dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
        config: &CompactionConfig,
    ) -> ContextObservation {
        let view = session.view();
        let estimated_tokens = tokens::estimate_context(&view.messages, tools, estimator) as u32;
        self.water.record_estimate(estimated_tokens);
        self.water.set_window(config.context_window_tokens);
        let preflight_overflow = self.water.exceeds_window() && !self.water.auto_compact_suppressed;
        ContextObservation {
            estimated_tokens,
            should_compact: self.water.should_compact(config) || preflight_overflow,
            preflight_overflow,
        }
    }

    /// Prepare messages before an LLM step through observe → commit → project.
    pub async fn prepare_step(
        &mut self,
        deps: &mut ContextDeps<'_>,
        tools: &[ToolDefinition],
    ) -> Result<PrepareStepResult> {
        let _ = deps.session.try_view().map_err(|reason| {
            anyhow!(
                "event-backed context projection unavailable: {}",
                reason.as_str()
            )
        })?;
        let observation = self.observe(deps.session, tools, deps.estimator, deps.compaction_config);
        self.capture_tail_sections(deps.hooks);
        let commit = self.commit(deps, tools, &observation).await?;
        let step = self.project(deps.session, tools, deps.estimator);
        let explain = self.explain_projection_for_step(deps.session, &step);
        self.remember_projection(&step, &explain);
        Ok(PrepareStepResult {
            step,
            compaction: commit.compaction,
            events: commit.events,
            explain,
        })
    }

    async fn commit(
        &mut self,
        deps: &mut ContextDeps<'_>,
        tools: &[ToolDefinition],
        observation: &ContextObservation,
    ) -> Result<CommitResult> {
        let mut events = Vec::new();
        self.flush_pending_publish(deps, &mut events).await?;
        self.ensure_initial_publish(deps, &mut events).await?;
        self.maybe_start_prefire(deps, tools);

        if !observation.should_compact {
            return Ok(CommitResult {
                compaction: None,
                events,
            });
        }
        if apply_steps_truncate_pass(deps.session, deps.compaction_config) {
            self.sync_water_from_estimate(
                deps.session,
                tools,
                deps.estimator,
                deps.compaction_config,
            );
            if !self.water.should_compact(deps.compaction_config) && !self.water.exceeds_window() {
                return Ok(CommitResult {
                    compaction: None,
                    events,
                });
            }
        }

        self.prefire.await_in_flight().await;
        let prefire_cache = self.prefire_cache_for_session(deps.session);
        self.maybe_flush_memory(deps, &mut events).await?;
        let memory_block = deps.handler.memory_reminder();
        Self::emit(
            deps,
            &mut events,
            ContextEvent::Checkpoint {
                reason: "pre_auto_compact",
            },
        )
        .await?;
        let reason = if observation.preflight_overflow {
            "preflight_overflow"
        } else {
            "token_threshold"
        };
        let compaction = match compact_session(
            deps.session,
            deps.client,
            deps.estimator,
            CompactionParams {
                model: deps.model,
                config: deps.compaction_config,
                reason,
                tools,
                options: CompactionOptions {
                    hooks: deps.hooks,
                    prefire: prefire_cache.as_ref(),
                    memory_block: memory_block.as_deref(),
                    ..Default::default()
                },
            },
        )
        .await
        {
            Ok(Some(result)) => {
                Self::emit_compaction_segments(deps, &mut events, &result).await?;
                self.prefire.clear();
                self.water.clear_auto_compact_suppression();
                self.bump_epoch_and_publish("compaction", deps, &mut events)
                    .await?;
                self.sync_water_from_estimate(
                    deps.session,
                    tools,
                    deps.estimator,
                    deps.compaction_config,
                );
                Self::emit(
                    deps,
                    &mut events,
                    ContextEvent::Checkpoint {
                        reason: "post_auto_compact",
                    },
                )
                .await?;
                Some(result)
            }
            Ok(None) => None,
            Err(err) => {
                warn!(error = %err, "auto-compact failed; suppressing until /compact");
                self.water.suppress_auto_compact();
                None
            }
        };
        deps.session.ensure_system_message(deps.system_prompt);
        Ok(CommitResult { compaction, events })
    }

    pub fn record_step_usage(
        &mut self,
        usage: &TokenUsage,
        session: &mut dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
        compaction_config: &CompactionConfig,
    ) -> Result<ContextUsageUpdate> {
        self.water.record_usage(usage);
        if let Some(gateway_hit) = usage.gateway_hit_tokens {
            self.last_gateway_hit_tokens = Some(gateway_hit);
            tracing::debug!(
                gateway_hit_tokens = gateway_hit,
                "inference gateway cache hits"
            );
        }
        if let Some(aligned) = usage.gateway_anchor_aligned {
            self.last_anchor_aligned = Some(aligned);
            tracing::debug!(
                anchor_aligned = aligned,
                "inference gateway anchor alignment"
            );
        }
        if let Some(cached) = usage.cached_tokens {
            let prefix = u64::from(self.last_prefix_tokens);
            self.last_cached_tokens = Some(cached);
            self.last_unchanged_reprocessed_est = Some(prefix.saturating_sub(cached));
            let effective = self.water.effective_tokens();
            if effective > 0 {
                tracing::info!(
                    cached_tokens = cached,
                    prompt_tokens = usage.prompt_tokens,
                    effective_tokens = effective,
                    unchanged_reprocessed_est = self.last_unchanged_reprocessed_est,
                    cache_pct = (cached * 100 / u64::from(effective.max(1))),
                    "provider prompt cache usage"
                );
            } else {
                tracing::debug!(cached_tokens = cached, "provider cache usage");
            }
        }
        let view = session.try_view().map_err(|reason| {
            anyhow!(
                "event-backed context projection unavailable: {}",
                reason.as_str()
            )
        })?;
        let estimated = tokens::estimate_context(&view.messages, tools, estimator) as u32;
        let context_tokens = self.water.usage_update(estimated);
        let context_window = compaction_config.context_window_tokens;
        session.update_context_usage(context_tokens, context_window);
        Ok(ContextUsageUpdate {
            context_tokens,
            context_window,
            context_percent: self.water.usage_percent(),
        })
    }

    pub async fn compact_forced(
        &mut self,
        deps: &mut ContextDeps<'_>,
        tools: &[ToolDefinition],
        user_hint: Option<&str>,
    ) -> Result<ForcedCompactResult> {
        let mut events = Vec::new();
        self.prefire.await_in_flight().await;
        let prefire_cache = self.prefire_cache_for_session(deps.session);
        self.maybe_flush_memory(deps, &mut events).await?;
        let memory_block = deps.handler.memory_reminder();
        Self::emit(
            deps,
            &mut events,
            ContextEvent::Checkpoint {
                reason: "pre_manual_compact",
            },
        )
        .await?;
        let result = compact_session_forced(
            deps.session,
            deps.client,
            deps.estimator,
            CompactionParams {
                model: deps.model,
                config: deps.compaction_config,
                reason: "manual",
                tools,
                options: CompactionOptions {
                    hooks: deps.hooks,
                    prefire: prefire_cache.as_ref(),
                    memory_block: memory_block.as_deref(),
                    force_summarize: true,
                    user_hint,
                },
            },
        )
        .await;
        self.prefire.clear();
        match result {
            Ok(compaction) => {
                self.water.clear_auto_compact_suppression();
                if compaction.is_some() {
                    if let Some(ref result) = compaction {
                        Self::emit_compaction_segments(deps, &mut events, result).await?;
                    }
                    self.bump_epoch_and_publish("manual_compaction", deps, &mut events)
                        .await?;
                    self.sync_water_from_estimate(
                        deps.session,
                        tools,
                        deps.estimator,
                        deps.compaction_config,
                    );
                    Self::emit(
                        deps,
                        &mut events,
                        ContextEvent::Checkpoint {
                            reason: "post_manual_compact",
                        },
                    )
                    .await?;
                }
                deps.session.ensure_system_message(deps.system_prompt);
                Ok(ForcedCompactResult { compaction, events })
            }
            Err(err) => {
                self.water.suppress_auto_compact();
                Err(err)
            }
        }
    }

    /// Handle provider context-overflow: steps-first tail truncate, then full compact.
    pub async fn handle_overflow(
        &mut self,
        deps: &mut ContextDeps<'_>,
        tools: &[ToolDefinition],
        overflow_truncated: &mut bool,
        overflow_summarized: &mut bool,
    ) -> Result<OverflowHandleResult> {
        let mut events = Vec::new();
        self.capture_tail_sections(deps.hooks);
        if !*overflow_truncated {
            *overflow_truncated = true;
            let mut steps_config = deps.compaction_config.clone();
            steps_config.intra_steps_first = true;
            if apply_steps_truncate_pass(deps.session, &steps_config) {
                info!("context overflow: applied steps-first truncate before retry");
                deps.session.ensure_system_message(deps.system_prompt);
                return Ok(OverflowHandleResult {
                    retry: true,
                    compaction: None,
                    events,
                });
            }
            info!(
                "context overflow: no current-turn tool tail to truncate; compacting instead of rewriting older bodies"
            );
        }
        if !*overflow_summarized {
            *overflow_summarized = true;
            self.prefire.await_in_flight().await;
            let prefire_cache = self.prefire_cache_for_session(deps.session);
            self.maybe_flush_memory(deps, &mut events).await?;
            let memory_block = deps.handler.memory_reminder();
            Self::emit(
                deps,
                &mut events,
                ContextEvent::Checkpoint {
                    reason: "pre_overflow_compact",
                },
            )
            .await?;
            match compact_session(
                deps.session,
                deps.client,
                deps.estimator,
                CompactionParams {
                    model: deps.model,
                    config: deps.compaction_config,
                    reason: "context_overflow",
                    tools,
                    options: CompactionOptions {
                        hooks: deps.hooks,
                        prefire: prefire_cache.as_ref(),
                        memory_block: memory_block.as_deref(),
                        ..Default::default()
                    },
                },
            )
            .await
            {
                Ok(Some(result)) => {
                    Self::emit_compaction_segments(deps, &mut events, &result).await?;
                    self.prefire.clear();
                    self.water.clear_auto_compact_suppression();
                    self.bump_epoch_and_publish("overflow_compaction", deps, &mut events)
                        .await?;
                    self.sync_water_from_estimate(
                        deps.session,
                        tools,
                        deps.estimator,
                        deps.compaction_config,
                    );
                    Self::emit(
                        deps,
                        &mut events,
                        ContextEvent::Checkpoint {
                            reason: "post_overflow_compact",
                        },
                    )
                    .await?;
                    deps.session.ensure_system_message(deps.system_prompt);
                    return Ok(OverflowHandleResult {
                        retry: true,
                        compaction: Some(result),
                        events,
                    });
                }
                Ok(None) => {
                    deps.session.ensure_system_message(deps.system_prompt);
                    return Ok(OverflowHandleResult {
                        retry: true,
                        compaction: None,
                        events,
                    });
                }
                Err(err) => {
                    warn!(error = %err, "overflow compact failed");
                    self.water.suppress_auto_compact();
                    return Err(err);
                }
            }
        }
        Ok(OverflowHandleResult {
            retry: false,
            compaction: None,
            events,
        })
    }

    pub fn is_context_overflow_error(err: &anyhow::Error) -> bool {
        is_context_overflow_error(err)
    }

    /// Explain the current outbound projection without mutating the session.
    pub fn explain_projection(
        &self,
        session: &dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
    ) -> ProjectionExplain {
        let step = self.project(session, tools, estimator);
        self.explain_projection_for_step(session, &step)
    }

    /// Strict event-backed explain variant for new integrations.
    pub fn try_explain_projection(
        &self,
        session: &dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
    ) -> Result<ProjectionExplain> {
        let _ = session.try_view().map_err(|reason| {
            anyhow!(
                "event-backed context projection unavailable: {}",
                reason.as_str()
            )
        })?;
        Ok(self.explain_projection(session, tools, estimator))
    }

    fn explain_projection_for_step(
        &self,
        session: &dyn ContextSession,
        step: &StepContext,
    ) -> ProjectionExplain {
        let view = session.view();
        let mut retained_turn_ids = Vec::new();
        for event in &view.active_events {
            let turn_id = match event {
                zene_session::SessionEvent::TurnStarted { turn_id, .. }
                | zene_session::SessionEvent::StepStarted { turn_id, .. }
                | zene_session::SessionEvent::TurnEnded { turn_id, .. }
                | zene_session::SessionEvent::Checkpoint {
                    turn_id: Some(turn_id),
                    ..
                }
                | zene_session::SessionEvent::ToolCall {
                    turn_id: Some(turn_id),
                    ..
                }
                | zene_session::SessionEvent::ToolResult {
                    turn_id: Some(turn_id),
                    ..
                }
                | zene_session::SessionEvent::PermissionDecision {
                    turn_id: Some(turn_id),
                    ..
                } => turn_id,
                _ => continue,
            };
            if !retained_turn_ids.iter().any(|id| id == turn_id) {
                retained_turn_ids.push(turn_id.clone());
            }
        }
        let retained_turn_count = retained_turn_ids.len();
        let compaction_event_ids = view
            .active_events
            .iter()
            .filter_map(|event| match event {
                zene_session::SessionEvent::CompactionApplied { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        ProjectionExplain {
            source_message_count: view.messages.len(),
            projected_message_count: step.messages.len(),
            source_event_count: view.source_event_count,
            active_event_count: view.active_events.len(),
            cache_drift_detected: view.cache_drift_detected,
            used_materialized_fallback: view.used_materialized_fallback,
            fallback_reason: view
                .fallback_reason
                .map(|reason| reason.as_str().to_string()),
            active_branch_id: view.active_branch_id,
            active_path_start_sequence: view.active_path_start_sequence,
            injected: projection_injected_labels(&step.messages),
            retained_message_count: step.messages.len(),
            retained_turn_count,
            dropped_event_count: view
                .source_event_count
                .saturating_sub(view.active_events.len()),
            truncated_message_count: projection_truncated_message_count(&step.messages),
            compaction_event_ids,
            tool_output_provenance: projection_tool_output_provenance(&step.messages),
            retained_turn_ids,
            injected_sources: projection_injected_sources(&step.messages),
            delivery: match step.metadata.delivery {
                zene_llm::ContextDelivery::Full => DeliveryMode::Full,
                zene_llm::ContextDelivery::Delta => DeliveryMode::Delta,
            },
            delivery_tail_start: step.metadata.tail_start,
            estimate_tokens: step.estimate_tokens,
            context_epoch: step.metadata.context_epoch,
            prefix_cache: self.prefix_cache_explain(step),
        }
    }

    fn prefix_cache_explain(&self, step: &StepContext) -> PrefixCacheExplain {
        let layout = split_layout(&step.messages);
        let fingerprint = prefix_fingerprint(&step.messages, layout.prefix_end);
        let epoch_reason = self.last_epoch_bump_reason;
        let epoch_bumped = epoch_reason.is_some();
        let break_kind = classify_prefix_break(
            self.last_prefix_fingerprint.as_deref(),
            fingerprint.as_deref(),
            epoch_bumped,
            epoch_reason,
        );
        PrefixCacheExplain {
            prefix_end: layout.prefix_end,
            body_end: layout.body_end,
            tail_decoration_count: layout.tail_decoration_count,
            prefix_fingerprint: fingerprint,
            break_kind: break_kind.as_str().to_string(),
            cached_tokens: self.last_cached_tokens,
            gateway_hit_tokens: self.last_gateway_hit_tokens,
            anchor_aligned: self.last_anchor_aligned,
            unchanged_reprocessed_est: self.last_unchanged_reprocessed_est,
        }
    }

    fn capture_tail_sections(&mut self, hooks: Option<&dyn ContextHooks>) {
        if let Some(hooks) = hooks {
            let mut sections = hooks.step_tail_decorations();
            let mut extra = hooks.mutate_tail_decorations(self.epoch);
            sections.append(&mut extra);
            self.tail_sections = sections;
        }
    }

    fn remember_projection(&mut self, step: &StepContext, explain: &ProjectionExplain) {
        self.last_prefix_fingerprint = explain.prefix_cache.prefix_fingerprint.clone();
        let prefix_end = explain.prefix_cache.prefix_end.min(step.messages.len());
        self.last_prefix_tokens = if prefix_end == 0 {
            0
        } else {
            // Cheap standing estimate: chars/4 of the frozen prefix.
            let chars: usize = step.messages[..prefix_end]
                .iter()
                .map(|message| message.content.as_deref().unwrap_or("").chars().count())
                .sum();
            u32::try_from((chars / 4).max(1)).unwrap_or(u32::MAX)
        };
        self.last_epoch_bump_reason = None;
    }

    fn project(
        &self,
        session: &dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
    ) -> StepContext {
        let mode = delivery_mode_from_env();
        let view = session.view();
        let mut messages = view.messages;
        relocate_prefix_adjacent_decorations(&mut messages);
        apply_tail_decorations(&mut messages, &self.tail_sections);
        debug_assert!(
            prefix_adjacent_decoration_index(&messages).is_none(),
            "live decoration must not sit between pinned prefix and conversation body"
        );
        let assembled = assemble_outbound(&messages, self.gateway_prefix_len, mode);
        let metadata = self.metadata_for_outbound(session, &assembled);
        let estimate_tokens =
            tokens::estimate_context(&assembled.messages, tools, estimator) as u32;
        StepContext {
            messages: assembled.messages,
            metadata,
            estimate_tokens,
        }
    }

    fn sync_water_from_estimate(
        &mut self,
        session: &mut dyn ContextSession,
        tools: &[ToolDefinition],
        estimator: &TokenEstimator,
        compaction_config: &CompactionConfig,
    ) {
        let view = session.view();
        let estimated = tokens::estimate_context(&view.messages, tools, estimator) as u32;
        self.water.record_estimate(estimated);
        self.water.last_prompt_tokens = Some(estimated);
        session.update_context_usage(estimated, compaction_config.context_window_tokens);
    }

    fn session_id_for(&self, session: &dyn ContextSession) -> String {
        self.external_session_id
            .clone()
            .unwrap_or_else(|| session.session_id().to_string())
    }

    async fn emit(
        deps: &mut ContextDeps<'_>,
        events: &mut Vec<ContextEvent>,
        event: ContextEvent,
    ) -> Result<EventOutcome> {
        if let ContextEvent::Checkpoint { reason } = &event {
            deps.session.persist_checkpoint(reason)?;
            events.push(event);
            return Ok(EventOutcome::Void);
        }
        let outcome = deps.handler.handle(&event).await?;
        events.push(event);
        Ok(outcome)
    }

    async fn emit_compaction_segments(
        deps: &mut ContextDeps<'_>,
        events: &mut Vec<ContextEvent>,
        result: &CompactionResult,
    ) -> Result<()> {
        if let Some(segment) = &result.segment {
            Self::emit(
                deps,
                events,
                ContextEvent::CompactionSegment {
                    session_id: segment.session_id.clone(),
                    body: segment.body.clone(),
                },
            )
            .await?;
        }
        Ok(())
    }

    async fn flush_pending_publish(
        &mut self,
        deps: &mut ContextDeps<'_>,
        events: &mut Vec<ContextEvent>,
    ) -> Result<()> {
        if !self.pending_publish {
            return Ok(());
        }
        self.pending_publish = false;
        let view = deps.session.view();
        self.gateway_prefix_len = view.messages.len();
        let session_id = self.session_id_for(deps.session);
        Self::emit(
            deps,
            events,
            ContextEvent::PublishPrefix {
                session_id,
                epoch: self.epoch,
                messages: view.messages,
            },
        )
        .await?;
        Ok(())
    }

    async fn ensure_initial_publish(
        &mut self,
        deps: &mut ContextDeps<'_>,
        events: &mut Vec<ContextEvent>,
    ) -> Result<()> {
        if self.initial_publish_done || !gateway_configured() {
            return Ok(());
        }
        self.initial_publish_done = true;
        let view = deps.session.view();
        self.gateway_prefix_len = view.messages.len();
        let session_id = self.session_id_for(deps.session);
        let pinned_boundary = stable_system_boundary(&view.messages);
        info!(
            session_id = %session_id,
            epoch = self.epoch,
            messages = view.messages.len(),
            pinned_boundary,
            "initial gateway prefix publish"
        );
        Self::emit(
            deps,
            events,
            ContextEvent::PublishPrefix {
                session_id,
                epoch: self.epoch,
                messages: view.messages,
            },
        )
        .await?;
        Ok(())
    }

    async fn bump_epoch_and_publish(
        &mut self,
        reason: &'static str,
        deps: &mut ContextDeps<'_>,
        events: &mut Vec<ContextEvent>,
    ) -> Result<()> {
        let old = self.epoch;
        self.epoch = self.epoch.saturating_add(1);
        self.last_epoch_bump_reason = Some(reason);
        let view = deps.session.view();
        self.gateway_prefix_len = view.messages.len();
        info!(
            old,
            new = self.epoch,
            reason,
            gateway_prefix_len = self.gateway_prefix_len,
            "context epoch bumped"
        );
        let session_id = self.session_id_for(deps.session);
        Self::emit(
            deps,
            events,
            ContextEvent::PublishPrefix {
                session_id,
                epoch: self.epoch,
                messages: view.messages,
            },
        )
        .await?;
        Ok(())
    }

    fn prefire_cache_for_session(&self, session: &dyn ContextSession) -> Option<PrefireCache> {
        let view = session.view();
        let prefix_start = if view
            .messages
            .first()
            .is_some_and(|m| m.role == zene_llm::Role::System)
        {
            1
        } else {
            0
        };
        let body = &view.messages[prefix_start..];
        self.prefire.valid_cache_for(body)
    }

    fn maybe_start_prefire(&self, deps: &ContextDeps<'_>, _tools: &[ToolDefinition]) {
        #[cfg(not(feature = "prefire"))]
        let _ = deps;
        #[cfg(feature = "prefire")]
        {
            let Some(factory) = deps.prefire_client_factory.as_ref() else {
                return;
            };
            let lead = prefire::prefire_lead_percent();
            if !self.water.should_prefire(deps.compaction_config, lead) {
                return;
            }
            if self.prefire.is_in_flight() || self.prefire.has_cache() {
                return;
            }

            let messages = deps.session.view().messages;
            let prefix_start = if messages
                .first()
                .is_some_and(|m| m.role == zene_llm::Role::System)
            {
                1
            } else {
                0
            };
            if messages.len().saturating_sub(prefix_start) < 8 {
                return;
            }
            let body = messages[prefix_start..].to_vec();
            let split = two_pass::split_messages_for_two_pass(
                &body,
                deps.estimator,
                two_pass::TWO_PASS_DEFAULT_SPLIT_FRACTION,
            );
            if split.split_idx == 0 || split.split_idx >= body.len() {
                return;
            }
            let pass1_prefix = body[..split.split_idx].to_vec();
            let fingerprint = two_pass::fingerprint_messages(&pass1_prefix);
            if self.prefire.already_launched_for(fingerprint) {
                return;
            }

            let factory = factory.clone();
            let model = deps.model.to_string();
            let window = deps.compaction_config.context_window_tokens;
            let split_idx = split.split_idx;
            let estimator = *deps.estimator;
            info!(
                split_idx,
                usage_percent = self.water.usage_percent(),
                "starting prefire pass1"
            );
            let handle = tokio::spawn(async move {
                let client = match factory().await {
                    Ok(c) => c,
                    Err(err) => {
                        warn!(error = %err, "prefire: failed to create client");
                        return None;
                    }
                };
                match crate::compaction::summarize_messages_with_ladder(
                    &client,
                    &model,
                    &pass1_prefix,
                    Some(window),
                    &estimator,
                )
                .await
                {
                    Ok(note1) => {
                        if note1.trim().is_empty() {
                            return None;
                        }
                        info!(note1_chars = note1.len(), "prefire pass1 cached NOTE₁");
                        Some(PrefireCache {
                            note1,
                            fingerprint,
                            split_idx,
                        })
                    }
                    Err(err) => {
                        warn!(error = %err, "prefire pass1 failed");
                        None
                    }
                }
            });
            self.prefire.set_handle(fingerprint, handle);
        }
    }

    async fn maybe_flush_memory(
        &mut self,
        deps: &mut ContextDeps<'_>,
        events: &mut Vec<ContextEvent>,
    ) -> Result<()> {
        let cycle = deps.session.compaction_cycle();
        let marker = cycle.saturating_add(1);
        let threshold = ContextWaterLevel::auto_compact_threshold_percent(deps.compaction_config);
        if !memory::should_flush(
            self.water.usage_percent(),
            threshold,
            self.last_memory_flush_compaction == marker,
        ) {
            return Ok(());
        }
        let conversation = memory::format_flush_input(&deps.session.view().messages);
        if conversation.trim().is_empty() {
            return Ok(());
        }
        let outcome = Self::emit(deps, events, ContextEvent::MemoryFlush { conversation }).await?;
        if let EventOutcome::MemoryFlush(memory::FlushResult::Accepted) = outcome {
            self.last_memory_flush_compaction = marker;
        }
        Ok(())
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;
    use zene_llm::Message;

    #[test]
    fn detects_tool_truncation_and_handle_markers() {
        let messages = vec![
            Message::tool_result(
                "call-1",
                "Read",
                "preview\n\n[truncated 42 bytes; full output saved to /tmp/out.txt.]",
            ),
            Message::tool_result(
                "call-2",
                "Bash",
                "[zene-tool-output path=\"/tmp/handle.txt\" bytes=100]",
            ),
        ];
        let provenance = projection_tool_output_provenance(&messages);
        assert_eq!(provenance.len(), 2);
        assert_eq!(provenance[0].kind, "truncated");
        assert_eq!(
            provenance[0].handle_reference.as_deref(),
            Some("/tmp/out.txt")
        );
        assert_eq!(provenance[1].kind, "handle");
        assert_eq!(
            provenance[1].handle_reference.as_deref(),
            Some("/tmp/handle.txt")
        );
    }

    #[test]
    fn classifies_injected_sources() {
        let messages = vec![
            Message::compaction_summary("summary"),
            Message::user("<system-reminder>\n<memory-context>memory</memory-context>\nActive todos\n</system-reminder>"),
        ];
        let sources = projection_injected_sources(&messages);
        assert_eq!(sources[0].source, "compaction_event");
        assert_eq!(sources[1].source, "memory");
        assert_eq!(sources[2].source, "todos");
    }

    #[test]
    fn tail_decorations_keep_prefix_fingerprint_stable() {
        use crate::tokens::TokenEstimator;
        use zene_session::SessionRecord;

        let mut engine = ContextEngine::new(128_000);
        let mut session = SessionRecord::new(std::path::Path::new("/tmp"));
        session.ensure_system_message("frozen system");
        session.push_message(Message::user("hello"));
        session.push_message(Message::assistant("hi"));
        let estimator = TokenEstimator::default();
        let tools = [];

        let first = engine.assemble_step(&session, &tools, &estimator);
        let first_prefix = crate::layout::prefix_fingerprint(&first.messages, 1);
        let explain_first = engine.explain_projection(&session, &tools, &estimator);
        assert_eq!(explain_first.prefix_cache.prefix_end, 1);
        assert_eq!(explain_first.prefix_cache.tail_decoration_count, 0);

        engine.set_step_tail_decorations(vec!["Active todos:\n- [pending] ship".into()]);
        let second = engine.assemble_step(&session, &tools, &estimator);
        let explain_second = engine.explain_projection(&session, &tools, &estimator);
        assert_eq!(explain_second.prefix_cache.prefix_end, 1);
        assert_eq!(explain_second.prefix_cache.tail_decoration_count, 1);
        assert!(second
            .messages
            .last()
            .unwrap()
            .content
            .as_deref()
            .unwrap()
            .contains("ship"));
        assert_eq!(
            first_prefix,
            crate::layout::prefix_fingerprint(&second.messages, 1)
        );

        engine.set_step_tail_decorations(vec!["You are in Plan mode.".into()]);
        let third = engine.assemble_step(&session, &tools, &estimator);
        assert_eq!(
            first_prefix,
            crate::layout::prefix_fingerprint(&third.messages, 1)
        );
        assert_eq!(
            engine
                .explain_projection(&session, &tools, &estimator)
                .prefix_cache
                .break_kind,
            "none"
        );
    }

    #[test]
    fn assemble_relocates_prefix_adjacent_index_block() {
        use crate::tokens::TokenEstimator;
        use zene_session::SessionRecord;

        let mut engine = ContextEngine::new(128_000);
        let mut session = SessionRecord::new(std::path::Path::new("/tmp"));
        session.ensure_system_message("frozen system");
        session.push_message(Message::user(
            "<system-reminder>\n<agent_documents_index>\n</system-reminder>",
        ));
        session.push_message(Message::user("hello"));
        let estimator = TokenEstimator::default();
        let tools = [];
        engine.set_step_tail_decorations(vec!["live tail".into()]);
        let step = engine.assemble_step(&session, &tools, &estimator);
        assert!(crate::layout::prefix_adjacent_decoration_index(&step.messages).is_none());
        assert_eq!(step.messages[1].content.as_deref(), Some("hello"));
        assert!(step
            .messages
            .last()
            .unwrap()
            .content
            .as_deref()
            .unwrap()
            .contains("live tail"));
        assert!(!step.messages.iter().any(|message| message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("agent_documents_index"))));
    }

    #[test]
    fn system_resize_is_distinct_from_append_only_none() {
        use crate::tokens::TokenEstimator;
        use zene_session::SessionRecord;

        let mut engine = ContextEngine::new(128_000);
        let mut session = SessionRecord::new(std::path::Path::new("/tmp"));
        session.ensure_system_message("sys v1");
        session.push_message(Message::user("hello"));
        let estimator = TokenEstimator::default();
        let tools = [];
        let _ = engine.assemble_step(&session, &tools, &estimator);

        session.push_message(Message::assistant("hi"));
        let append = engine.explain_projection(&session, &tools, &estimator);
        assert_eq!(append.prefix_cache.break_kind, "none");

        session.update_system_prefix("sys v2 is longer and breaks the prefix");
        let _ = engine.on_system_prefix_changed("system_prefix");
        let resized = engine.explain_projection(&session, &tools, &estimator);
        assert_eq!(resized.prefix_cache.break_kind, "system_resize");
    }
}
