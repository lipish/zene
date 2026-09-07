//! Workspace context for system prompts.

mod prompt;
mod provider;
mod skills;

pub use prompt::{build_system_prompt, build_system_prompt_ext, CLOUD_GITHUB_RULE};
pub use provider::{FsWorkspaceProvider, WorkspaceProvider};
pub use skills::{format_available_skills, SkillMeta};
