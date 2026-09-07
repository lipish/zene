use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use zene_llm::ToolDefinition;

use crate::line_endings::{make_carriage_returns_visible, to_model_text_view, LineEndingStyle};
use crate::registry::{Tool, ToolContext, ToolResult};

pub struct ReadTool;

const MAX_READ_BYTES: usize = 50 * 1024;
const MAX_READ_LINES: usize = 2000;

#[derive(Debug, Deserialize)]
struct ReadArgs {
    path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "Read"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "Read".to_string(),
            description: "Read a file or list a directory. Output is truncated to 2000 lines or 50KB — use offset/limit for large files.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Relative or absolute file or directory path" },
                    "offset": { "type": "integer", "description": "1-based start line" },
                    "limit": { "type": "integer", "description": "Maximum number of lines to read" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, arguments: &str, ctx: &ToolContext) -> Result<ToolResult> {
        let args: ReadArgs = serde_json::from_str(arguments).context("parse Read args")?;
        match format_read(&args, ctx).await {
            Ok(content) => Ok(ToolResult {
                content,
                is_error: false,
            }),
            Err(err) => Ok(ToolResult {
                content: err.to_string(),
                is_error: true,
            }),
        }
    }
}

async fn format_read(args: &ReadArgs, ctx: &ToolContext) -> Result<String> {
    let resolved = ctx.sandbox.resolve(&args.path)?;

    if resolved.is_dir() {
        return list_directory(&resolved);
    }

    let raw = ctx
        .sandbox
        .read_file_bytes(&args.path, MAX_READ_BYTES + 1)
        .await?;
    if zene_sandbox::is_binary_content(&raw) {
        anyhow::bail!(
            "cannot read binary file: {}. Read supports text files only.",
            args.path
        );
    }
    let raw_str = String::from_utf8(raw).context("file is not valid UTF-8 text")?;
    let view = to_model_text_view(&raw_str);
    let text = if view.line_ending_style == LineEndingStyle::Mixed {
        make_carriage_returns_visible(&view.text)
    } else {
        view.text
    };
    format_file_content(&args.path, &text, args.offset, args.limit)
}

fn list_directory(resolved: &std::path::Path) -> Result<String> {
    let mut entries: Vec<String> = std::fs::read_dir(resolved)
        .with_context(|| format!("read directory: {}", resolved.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    Ok(if entries.is_empty() {
        "(empty directory)".to_string()
    } else {
        entries.join("\n")
    })
}

fn format_file_content(
    _path: &str,
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<String> {
    let all_lines: Vec<&str> = content.lines().collect();
    let start_line = offset.unwrap_or(1).saturating_sub(1);
    if start_line >= all_lines.len() && !all_lines.is_empty() {
        anyhow::bail!(
            "offset {} is beyond end of file ({} lines total)",
            offset.unwrap_or(1),
            all_lines.len()
        );
    }
    if start_line >= all_lines.len() {
        return Ok(String::new());
    }

    let end_line = match limit {
        Some(lim) => (start_line + lim).min(all_lines.len()),
        None => all_lines.len(),
    };
    let slice = &all_lines[start_line..end_line];

    let (truncated, was_truncated) = truncate_head(slice, MAX_READ_LINES, MAX_READ_BYTES);
    let numbered: Vec<String> = truncated
        .lines()
        .enumerate()
        .map(|(idx, line)| format!("L{}:{}", start_line + idx + 1, line))
        .collect();
    let mut output = numbered.join("\n");

    if was_truncated {
        let shown_end = start_line + truncated.lines().count();
        output.push_str(&format!(
            "\n\n[Showing lines {}-{} of {}. Use offset={} to continue.]",
            start_line + 1,
            shown_end,
            all_lines.len(),
            shown_end + 1
        ));
    }

    if output.is_empty() && all_lines.is_empty() {
        return Ok(String::new());
    }

    Ok(output)
}

fn truncate_head(lines: &[&str], max_lines: usize, max_bytes: usize) -> (String, bool) {
    let mut result = String::new();
    let mut was_truncated = false;
    for (line_count, line) in lines.iter().enumerate() {
        if line_count >= max_lines {
            was_truncated = true;
            break;
        }
        let next = if line_count == 0 {
            line.to_string()
        } else {
            format!("\n{}", line)
        };
        if result.len() + next.len() > max_bytes {
            was_truncated = true;
            break;
        }
        result.push_str(&next);
    }

    (result, was_truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_byte_limit() {
        let lines: Vec<&str> = vec!["a"; 100];
        let (text, truncated) = truncate_head(&lines, 2000, 10);
        assert!(truncated);
        assert!(text.len() <= 10);
    }

    #[test]
    fn truncate_respects_line_limit() {
        let lines: Vec<String> = (0..100).map(|i| format!("line{i}")).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let (_, truncated) = truncate_head(&refs, 5, 50 * 1024);
        assert!(truncated);
    }
}
