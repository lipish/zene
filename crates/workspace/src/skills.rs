use std::path::{Path, PathBuf};

/// Metadata for a discovered skill (`SKILL.md`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

/// Parse YAML-style frontmatter for `name` and `description` only.
pub fn skill_meta_from_file(path: &Path, content: &str) -> Option<SkillMeta> {
    let frontmatter = parse_frontmatter(content)?;
    let fallback_name = path
        .parent()
        .and_then(|dir| dir.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string());

    let name = frontmatter
        .get("name")
        .cloned()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(fallback_name);

    let description = frontmatter.get("description").cloned().unwrap_or_default();

    Some(SkillMeta {
        name,
        description,
        path: path.to_path_buf(),
    })
}

pub fn format_available_skills(skills: &[SkillMeta]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }

    let lines: Vec<String> = skills
        .iter()
        .map(|skill| format!("- {}: {}", skill.name, skill.description))
        .collect();
    Some(format!("Available skills:\n{}", lines.join("\n")))
}

fn parse_frontmatter(content: &str) -> Option<std::collections::HashMap<String, String>> {
    let (frontmatter, _) = split_frontmatter(content)?;
    let mut map = std::collections::HashMap::new();
    for line in frontmatter.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        map.insert(key.trim().to_lowercase(), value.trim().to_string());
    }
    Some(map)
}

fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let trimmed = content.strip_prefix("---")?;
    let rest = trimmed.strip_prefix('\n').or(Some(trimmed))?;
    let end = rest.find("\n---")?;
    let frontmatter = &rest[..end];
    let body = rest[end + 4..]
        .strip_prefix('\n')
        .unwrap_or(&rest[end + 4..]);
    Some((frontmatter, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_available_skills_lists_entries() {
        let formatted = format_available_skills(&[SkillMeta {
            name: "demo".to_string(),
            description: "Demo skill".to_string(),
            path: PathBuf::from(".agents/skills/demo/SKILL.md"),
        }])
        .unwrap();
        assert!(formatted.contains("Available skills:"));
        assert!(formatted.contains("- demo: Demo skill"));
    }

    #[test]
    fn parses_skill_frontmatter() {
        let meta = skill_meta_from_file(
            Path::new(".agents/skills/demo/SKILL.md"),
            "---\nname: demo\ndescription: A demo skill\n---\n\n# Demo\n",
        )
        .unwrap();
        assert_eq!(meta.name, "demo");
        assert_eq!(meta.description, "A demo skill");
    }
}
