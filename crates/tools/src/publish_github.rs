use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use zene_llm::ToolDefinition;

use crate::registry::{Tool, ToolContext, ToolResult};

const TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Deserialize)]
struct PublishArgs {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    draft: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct CloudPublishConfig {
    pub api_url: String,
    pub token: String,
    pub run_id: String,
}

impl CloudPublishConfig {
    pub fn from_env() -> Option<Self> {
        let run_id = std::env::var("ZENE_RUN_ID")
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        let api_url = std::env::var("ZENE_CLOUD_API_URL")
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        let token = std::env::var("ZENE_CLOUD_WORKER_TOKEN")
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        Some(Self {
            api_url: api_url.trim_end_matches('/').to_string(),
            token,
            run_id,
        })
    }
}

pub struct PublishGithubTool {
    config: Option<CloudPublishConfig>,
}

impl PublishGithubTool {
    pub fn new() -> Self {
        Self {
            config: CloudPublishConfig::from_env(),
        }
    }

    pub fn with_config(config: CloudPublishConfig) -> Self {
        Self {
            config: Some(config),
        }
    }

    pub fn available() -> bool {
        CloudPublishConfig::from_env().is_some()
    }

    pub fn is_configured(&self) -> bool {
        self.config.is_some()
    }
}

impl Default for PublishGithubTool {
    fn default() -> Self {
        Self::new()
    }
}

fn format_publish_result(push: &Value, pr: Option<&Value>) -> String {
    let sha = push
        .get("headSha")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let mut out = format!("Pushed session branch via Cloud git-broker (head {sha}).");
    if let Some(pr) = pr {
        let title = pr.get("title").and_then(Value::as_str).unwrap_or("PR");
        if let Some(url) = pr.get("url").and_then(Value::as_str) {
            if let Some(n) = pr.get("providerNumber").and_then(Value::as_i64) {
                out.push_str(&format!(" Draft PR #{n}: {url} ({title})"));
            } else {
                out.push_str(&format!(" Draft PR: {url} ({title})"));
            }
        } else {
            out.push_str(&format!(" PR recorded: {title}"));
        }
    }
    out
}

#[async_trait]
impl Tool for PublishGithubTool {
    fn name(&self) -> &str {
        "PublishGithub"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "PublishGithub".to_string(),
            description: "Publish this Cloud session through the bound GitHub App: commit dirty workspace files if needed, push the session branch, and open a draft PR. Use this instead of git push, gh, or SSH-to-another-host publish.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Draft PR title. Defaults to the session title."
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional PR body."
                    },
                    "draft": {
                        "type": "boolean",
                        "description": "Open as draft (default true)."
                    }
                }
            }),
        }
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult> {
        let args: PublishArgs = if arguments.trim().is_empty() {
            PublishArgs {
                title: None,
                body: None,
                draft: None,
            }
        } else {
            serde_json::from_str(arguments).context("parse PublishGithub args")?
        };
        let Some(cfg) = self
            .config
            .as_ref()
            .cloned()
            .or_else(CloudPublishConfig::from_env)
        else {
            return Ok(ToolResult {
                content:
                    "PublishGithub is only available in a Zene Cloud session or when configured."
                        .into(),
                is_error: true,
            });
        };

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(TIMEOUT_SECS))
            .build()
            .context("publish http client")?;

        let push_url = format!("{}/internal/v1/runs/{}/git/push", cfg.api_url, cfg.run_id);
        let push_resp = client
            .post(&push_url)
            .bearer_auth(&cfg.token)
            .json(&json!({
                "force": false
            }))
            .send()
            .await
            .context("cloud git push")?;
        let push_status = push_resp.status();
        let push_text = push_resp.text().await.unwrap_or_default();
        if !push_status.is_success() {
            return Ok(ToolResult {
                content: format!("Cloud push failed ({push_status}): {push_text}"),
                is_error: true,
            });
        }
        let push_json: Value = serde_json::from_str(&push_text).unwrap_or(json!({}));

        let pr_url = format!(
            "{}/internal/v1/runs/{}/git/pull-request",
            cfg.api_url, cfg.run_id
        );
        let pr_resp = client
            .post(&pr_url)
            .bearer_auth(&cfg.token)
            .json(&json!({
                "title": args.title,
                "body": args.body,
                "draft": args.draft.unwrap_or(true),
            }))
            .send()
            .await
            .context("cloud create pull request")?;
        let pr_status = pr_resp.status();
        let pr_text = pr_resp.text().await.unwrap_or_default();
        if !pr_status.is_success() {
            return Ok(ToolResult {
                content: format!(
                    "{}\nCloud PR failed ({pr_status}): {pr_text}",
                    format_publish_result(&push_json, None)
                ),
                is_error: true,
            });
        }
        let pr_json: Value = serde_json::from_str(&pr_text).unwrap_or(json!({}));
        Ok(ToolResult {
            content: format_publish_result(&push_json, Some(&pr_json)),
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::format_publish_result;
    use serde_json::json;

    #[test]
    fn formats_push_and_pr() {
        let push = json!({ "headSha": "abc1234def" });
        let pr = json!({
            "title": "Fix login",
            "url": "https://github.com/acme/app/pull/9",
            "providerNumber": 9
        });
        let text = format_publish_result(&push, Some(&pr));
        assert!(text.contains("abc1234def"));
        assert!(text.contains("#9"));
        assert!(text.contains("https://github.com/acme/app/pull/9"));
    }
}
