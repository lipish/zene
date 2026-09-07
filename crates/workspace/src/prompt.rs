use crate::skills::format_available_skills;
use crate::WorkspaceProvider;

pub const CLOUD_GITHUB_RULE: &str = "\
# Cloud GitHub

This session is bound to a GitHub App installation. \
Call `PublishGithub` only when the user explicitly asks to commit, push, or open a pull request. \
Exploration, status checks, and “look at the project” must not publish. \
That tool is the only supported publish path: Cloud git-broker commits the workspace, pushes the session branch, and opens a draft PR. \
Do not run `git push` or `gh`. Do not SSH to another machine to publish. \
Local `git status`, `diff`, `log`, `add`, and `commit` in this workspace are allowed.";

/// Combine base system prompt with optional workspace context from a provider.
pub fn build_system_prompt(
    base: &str,
    provider: &dyn WorkspaceProvider,
    include_workspace_context: bool,
) -> String {
    build_system_prompt_ext(
        base,
        provider,
        include_workspace_context,
        std::env::var_os("ZENE_RUN_ID").is_some(),
    )
}

/// Combine base system prompt with optional workspace context and optional git publish rules.
pub fn build_system_prompt_ext(
    base: &str,
    provider: &dyn WorkspaceProvider,
    include_workspace_context: bool,
    include_github_rule: bool,
) -> String {
    let mut prompt = if !include_workspace_context {
        base.to_string()
    } else {
        let mut sections = vec![base.to_string()];

        if let Some(instructions) = provider.agent_instructions() {
            sections.push(format!("# Project instructions\n\n{instructions}"));
        }

        sections.push(provider.workspace_overview());

        if let Some(skills) = format_available_skills(&provider.discover_skills()) {
            sections.push(skills);
        }

        sections.join("\n\n")
    };
    if include_github_rule {
        prompt.push_str("\n\n");
        prompt.push_str(CLOUD_GITHUB_RULE);
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::FsWorkspaceProvider;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn skips_workspace_when_disabled() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("AGENTS.md"), "hidden").unwrap();
        let provider = FsWorkspaceProvider::new(dir.path());
        assert_eq!(
            build_system_prompt("base only", &provider, false),
            "base only"
        );
    }

    #[test]
    fn includes_sections_when_enabled() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("AGENTS.md"), "Use Rust.").unwrap();
        let provider = FsWorkspaceProvider::new(dir.path());
        let prompt = build_system_prompt("Base prompt.", &provider, true);
        assert!(prompt.starts_with("Base prompt."));
        assert!(prompt.contains("Project instructions"));
        assert!(prompt.contains("Use Rust."));
        assert!(prompt.contains("# Workspace"));
        assert!(!prompt.contains("Cloud GitHub"));

        let prompt_with_github = build_system_prompt_ext("Base prompt.", &provider, true, true);
        assert!(prompt_with_github.contains("Cloud GitHub"));
    }
}
