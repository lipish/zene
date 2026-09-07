use crate::ask_user::AskUserQuestionTool;
use crate::bash::BashTool;
use crate::edit::EditTool;
use crate::fetch_url::FetchUrlTool;
use crate::glob::GlobTool;
use crate::grep::GrepTool;
use crate::plan::{EnterPlanModeTool, ExitPlanModeTool};
use crate::publish_github::PublishGithubTool;
use crate::read::ReadTool;
use crate::registry::ToolRegistry;
use crate::repomap::RepoMapTool;
use crate::skill::SkillTool;
use crate::subagent::SubagentProfile;
use crate::task::TaskTool;
use crate::task_output::TaskOutputTool;
use crate::todo::{TodoListTool, TodoWriteTool};
use crate::web_search::WebSearchTool;
use crate::write::WriteTool;
use zene_config::{AgentProfile, WebSearchConfig};

pub fn builtin_tools(web_search: WebSearchConfig) -> ToolRegistry {
    ToolRegistry::new(all_builtin_tool_boxes(web_search))
}

pub fn agent_tools(profile: AgentProfile, web_search: WebSearchConfig) -> ToolRegistry {
    match profile {
        AgentProfile::Full => builtin_tools(web_search),
        AgentProfile::Explore => ToolRegistry::new(explore_agent_tool_boxes(web_search)),
        AgentProfile::Coder => ToolRegistry::new(coder_agent_tool_boxes(web_search)),
    }
}

pub fn default_builtin_tools() -> ToolRegistry {
    builtin_tools(WebSearchConfig::default())
}

pub fn tools_for_profile(profile: SubagentProfile) -> ToolRegistry {
    match profile {
        SubagentProfile::Explore => ToolRegistry::new(vec![
            Box::new(ReadTool),
            Box::new(GrepTool),
            Box::new(GlobTool),
            Box::new(RepoMapTool),
        ]),
        SubagentProfile::Coder => ToolRegistry::new(vec![
            Box::new(ReadTool),
            Box::new(WriteTool),
            Box::new(EditTool),
            Box::new(BashTool),
            Box::new(GrepTool),
            Box::new(GlobTool),
            Box::new(RepoMapTool),
        ]),
    }
}

fn with_cloud_publish(
    mut tools: Vec<Box<dyn crate::registry::Tool>>,
) -> Vec<Box<dyn crate::registry::Tool>> {
    if PublishGithubTool::available() {
        tools.push(Box::new(PublishGithubTool::new()));
    }
    tools
}

fn all_builtin_tool_boxes(web_search: WebSearchConfig) -> Vec<Box<dyn crate::registry::Tool>> {
    with_cloud_publish(vec![
        Box::new(ReadTool),
        Box::new(WriteTool),
        Box::new(EditTool),
        Box::new(BashTool),
        Box::new(GrepTool),
        Box::new(GlobTool),
        Box::new(RepoMapTool),
        Box::new(SkillTool),
        Box::new(TaskTool),
        Box::new(TaskOutputTool),
        Box::new(AskUserQuestionTool),
        Box::new(TodoWriteTool),
        Box::new(TodoListTool),
        Box::new(FetchUrlTool),
        Box::new(WebSearchTool::new(web_search)),
        Box::new(EnterPlanModeTool),
        Box::new(ExitPlanModeTool),
    ])
}

fn explore_agent_tool_boxes(web_search: WebSearchConfig) -> Vec<Box<dyn crate::registry::Tool>> {
    vec![
        Box::new(ReadTool),
        Box::new(GrepTool),
        Box::new(GlobTool),
        Box::new(RepoMapTool),
        Box::new(SkillTool),
        Box::new(AskUserQuestionTool),
        Box::new(TodoWriteTool),
        Box::new(TodoListTool),
        Box::new(FetchUrlTool),
        Box::new(WebSearchTool::new(web_search)),
        Box::new(EnterPlanModeTool),
        Box::new(ExitPlanModeTool),
    ]
}

fn coder_agent_tool_boxes(web_search: WebSearchConfig) -> Vec<Box<dyn crate::registry::Tool>> {
    with_cloud_publish(vec![
        Box::new(ReadTool),
        Box::new(WriteTool),
        Box::new(EditTool),
        Box::new(BashTool),
        Box::new(GrepTool),
        Box::new(GlobTool),
        Box::new(RepoMapTool),
        Box::new(SkillTool),
        Box::new(TaskTool),
        Box::new(TaskOutputTool),
        Box::new(AskUserQuestionTool),
        Box::new(TodoWriteTool),
        Box::new(TodoListTool),
        Box::new(FetchUrlTool),
        Box::new(WebSearchTool::new(web_search)),
        Box::new(EnterPlanModeTool),
        Box::new(ExitPlanModeTool),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use zene_config::WebSearchConfig;

    #[test]
    fn explore_profile_omits_write_tools() {
        let tools = agent_tools(AgentProfile::Explore, WebSearchConfig::default());
        let names: Vec<String> = tools.definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "Read"));
        assert!(names.iter().any(|n| n == "RepoMap"));
        assert!(!names.iter().any(|n| n == "Write"));
        assert!(!names.iter().any(|n| n == "Edit"));
        assert!(!names.iter().any(|n| n == "Bash"));
        assert!(!names.iter().any(|n| n == "Task"));
    }

    #[test]
    fn coder_profile_includes_write_and_task() {
        let tools = agent_tools(AgentProfile::Coder, WebSearchConfig::default());
        let names: Vec<String> = tools.definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "RepoMap"));
        assert!(names.iter().any(|n| n == "Write"));
        assert!(names.iter().any(|n| n == "Edit"));
        assert!(names.iter().any(|n| n == "Task"));
    }

    #[test]
    fn publish_github_is_absent_outside_cloud() {
        let names: Vec<String> = default_builtin_tools()
            .definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        if std::env::var_os("ZENE_RUN_ID").is_none()
            || std::env::var_os("ZENE_CLOUD_API_URL").is_none()
            || std::env::var_os("ZENE_CLOUD_WORKER_TOKEN").is_none()
        {
            assert!(!names.iter().any(|n| n == "PublishGithub"));
        }
    }
}
