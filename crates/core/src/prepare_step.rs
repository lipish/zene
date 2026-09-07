//! Main-agent context prepare-step extracted from [`crate::Agent`].
//!
//! Wave 14: `ContextAssemblerPort` still enters via Agent wiring, but todos
//! sync → ContextEngine prepare → ProjectionReady / water logging lives here
//! (parallel to [`crate::model_step`] for the model path).

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use zene_config::ZeneConfig;
use zene_context::{
    ContextDeps, ContextEngine, ContextModel, EstimateProvider, PrefireClientFactory,
    TokenEstimator,
};
use zene_llm::{ChatClient, ToolDefinition};
use zene_session::{AgentRecordWriter, RecordEntry, SessionRecord};
use zene_tools::{SharedBackgroundTasks, SharedTodoStore, ToolCatalog, ToolPolicy, ToolRegistry};
use zene_turn::PreparedContext;

use crate::context_config;
use crate::context_events::AgentContextHandler;
use crate::context_hooks::ZeneContextHooks;
use crate::events::{emit_event, AgentEvent};
use crate::plan_mode::tool_visible_in_definitions;
use crate::PromptOptions;

/// Mutable Agent pieces needed to assemble one step's [`PreparedContext`].
pub(crate) struct PrepareStepDeps<'a> {
    pub config: &'a ZeneConfig,
    pub context_model: &'a dyn ContextModel,
    pub context: &'a mut ContextEngine,
    pub session: &'a mut SessionRecord,
    pub system_prompt: &'a str,
    pub workdir: &'a Path,
    pub tools: &'a ToolRegistry,
    pub background: &'a SharedBackgroundTasks,
    pub todos: &'a SharedTodoStore,
    pub tool_policy: ToolPolicy,
    pub plan_mode_active: bool,
    pub record_writer: &'a AgentRecordWriter,
}

/// Assemble the next model-facing context (catalog defs + ContextEngine prepare).
pub(crate) async fn prepare_step_context(
    deps: PrepareStepDeps<'_>,
    options: &PromptOptions,
    cancel: Option<&CancellationToken>,
) -> Result<PreparedContext> {
    if check_cancelled(cancel)? {
        return Err(zene_turn::aborted_error());
    }

    sync_todos_to_session(deps.todos, deps.session);
    let plan_filter = deps.tool_policy.plan_mode && deps.plan_mode_active;
    let tools = tool_definitions_for_llm(deps.tools, plan_filter);
    let estimator = token_estimator(deps.config);
    let background_tasks = deps.background.lock().list();
    let hooks = ZeneContextHooks::new(deps.session, &background_tasks, plan_filter);
    let compaction_config = context_config::context_compaction_config(&deps.config.compaction);
    let mut handler =
        AgentContextHandler::new(deps.context_model, &deps.config.model, deps.workdir);
    let prefire_factory = prefire_client_factory(deps.config);
    let mut context_deps = ContextDeps {
        session: deps.session,
        compaction_config: &compaction_config,
        model: &deps.config.model,
        client: deps.context_model,
        hooks: Some(&hooks),
        system_prompt: deps.system_prompt,
        estimator: &estimator,
        handler: &mut handler,
        prefire_client_factory: prefire_factory,
    };
    let prepared = deps.context.prepare_step(&mut context_deps, &tools).await?;
    if let Some(result) = &prepared.compaction {
        record_compaction(deps.record_writer, result)?;
    }
    emit_event(
        &options.event_handler,
        AgentEvent::ProjectionReady(Box::new(
            crate::events::projection_ready_event_from_explain(&prepared.explain),
        )),
    );
    let step = prepared.step;
    debug!(
        estimated_context_tokens = step.estimate_tokens,
        effective_tokens = deps.context.water().effective_tokens(),
        usage_percent = deps.context.water().usage_percent(),
        message_count = step.messages.len(),
        tool_count = tools.len(),
        context_epoch = step.metadata.context_epoch,
        source_event_count = prepared.explain.source_event_count,
        projection_fallback = prepared.explain.used_materialized_fallback,
        "llm request context water level"
    );
    warn_if_near_context_limit(deps.config, step.estimate_tokens as usize);
    Ok(PreparedContext {
        messages: step.messages,
        tools,
        context_epoch: Some(step.metadata.context_epoch),
        metadata: Some(step.metadata),
        estimate_tokens: Some(step.estimate_tokens),
    })
}

fn tool_definitions_for_llm(tools: &ToolRegistry, plan_mode_active: bool) -> Vec<ToolDefinition> {
    ToolCatalog::definitions(tools)
        .into_iter()
        .filter(|def| tool_visible_in_definitions(&def.name, plan_mode_active))
        .collect()
}

fn sync_todos_to_session(todos: &SharedTodoStore, session: &mut SessionRecord) {
    let store = todos.lock();
    session.todos = store.to_items();
}

fn token_estimator(config: &ZeneConfig) -> TokenEstimator {
    TokenEstimator::for_provider(
        EstimateProvider::from_name(&config.provider),
        &config.model,
        config.chars_per_token_for_model(),
    )
}

fn prefire_client_factory(config: &ZeneConfig) -> Option<PrefireClientFactory> {
    let config = config.clone();
    Some(Arc::new(move || {
        let config = config.clone();
        Box::pin(async move {
            let client = ChatClient::from_config(&config).await?;
            Ok(Arc::new(client) as Arc<dyn ContextModel>)
        })
    }))
}

fn record_compaction(
    record_writer: &AgentRecordWriter,
    result: &zene_context::CompactionResult,
) -> Result<()> {
    record_writer.append(&RecordEntry::Compaction {
        reason: result.reason.clone(),
        compacted_count: result.compacted_count,
        tokens_before: Some(result.stats.tokens_before),
        tokens_after: Some(result.stats.tokens_after),
        ts: chrono::Utc::now(),
    })
}

fn warn_if_near_context_limit(config: &ZeneConfig, estimated_tokens: usize) {
    let window = config.compaction.context_window_tokens as f32;
    if window <= 0.0 {
        return;
    }
    let ratio = estimated_tokens as f32 / window;
    if ratio >= 0.9 {
        warn!(
            estimated_context_tokens = estimated_tokens,
            context_window_tokens = config.compaction.context_window_tokens,
            usage_ratio = ratio,
            "context estimate exceeds 90% of model window"
        );
    }
}

fn check_cancelled(cancel: Option<&CancellationToken>) -> Result<bool> {
    Ok(zene_turn::is_cancelled(cancel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zene_tools::default_builtin_tools;

    #[test]
    fn plan_mode_hides_write_tools_from_catalog_defs() {
        let tools = default_builtin_tools();
        let active = tool_definitions_for_llm(&tools, true);
        let names: Vec<_> = active.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Read"));
        assert!(!names
            .iter()
            .any(|n| *n == "Write" || *n == "Edit" || *n == "Bash"));
    }

    #[test]
    fn inactive_plan_mode_keeps_write_tools() {
        let tools = default_builtin_tools();
        let defs = tool_definitions_for_llm(&tools, false);
        let names: Vec<_> = defs.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Write"));
        assert!(names.contains(&"Bash"));
    }

    #[test]
    fn tool_policy_without_plan_mode_skips_filter_even_when_active_flag_set() {
        // Mirrors prepare_step: plan_filter = tool_policy.plan_mode && plan_mode_active
        let tools = default_builtin_tools();
        let policy = ToolPolicy::subagent();
        let plan_mode_active = true;
        let plan_filter = policy.plan_mode && plan_mode_active;
        let defs = tool_definitions_for_llm(&tools, plan_filter);
        assert!(defs.iter().any(|t| t.name == "Write"));
    }
}
