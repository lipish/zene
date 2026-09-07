//! Git safety policy for protected environments: prevents unintended remote pushes or publish actions.
//! Agents must not `git push` or invoke `gh` when Git push protection is enabled.

const DEFAULT_DENIED: &str = "\
Remote Git push or publish is restricted in this environment. \
Do not run `git push` or `gh`. Local `git status` / `diff` / `log` / `add` / `commit` is allowed. \
Keep changes in this workspace.";

pub fn is_cloud_run() -> bool {
    std::env::var_os("ZENE_RUN_ID").is_some()
}

pub fn deny_git_cli(command: &str) -> Option<String> {
    deny_git_cli_with_message(command, DEFAULT_DENIED)
}

pub fn deny_git_cli_with_message(command: &str, message: &str) -> Option<String> {
    blocked_kind(command).map(|kind| format!("Policy denied `{kind}`: {message}"))
}

pub fn deny_cloud_github_cli(command: &str) -> Option<String> {
    deny_git_cli(command)
}

fn blocked_kind(command: &str) -> Option<&'static str> {
    if git_subcommand_is(command, &["push", "send-pack", "request-pull"]) {
        return Some("git push");
    }
    if command_invokes("gh", command) {
        return Some("gh");
    }
    if command_invokes("hub", command) {
        return Some("hub");
    }
    None
}

fn command_invokes(bin: &str, command: &str) -> bool {
    for segment in split_segments(command) {
        if first_command(&segment).is_some_and(|name| name == bin) {
            return true;
        }
        if first_command(&segment).as_deref() == Some("ssh")
            && command_invokes(bin, &ssh_remote_command(&tokenize(&segment)[1..]))
        {
            return true;
        }
        if quoted_parts(&segment).any(|inner| command_invokes(bin, &inner)) {
            return true;
        }
    }
    false
}

fn git_subcommand_is(command: &str, blocked: &[&str]) -> bool {
    for segment in split_segments(command) {
        if let Some(sub) = git_subcommand(&segment) {
            if blocked.iter().any(|b| *b == sub) {
                return true;
            }
        }
        if quoted_parts(&segment).any(|inner| git_subcommand_is(&inner, blocked)) {
            return true;
        }
    }
    false
}

fn split_segments(command: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = command.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            cur.push(c);
            if c == q {
                quote = None;
            } else if c == '\\' {
                if let Some(n) = chars.next() {
                    cur.push(n);
                }
            }
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                cur.push(c);
            }
            ';' | '\n' => flush_seg(&mut out, &mut cur),
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                flush_seg(&mut out, &mut cur);
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                flush_seg(&mut out, &mut cur);
            }
            '|' => flush_seg(&mut out, &mut cur),
            _ => cur.push(c),
        }
    }
    flush_seg(&mut out, &mut cur);
    out
}

fn flush_seg(out: &mut Vec<String>, cur: &mut String) {
    let s = cur.trim().to_string();
    cur.clear();
    if !s.is_empty() {
        out.push(s);
    }
}

fn quoted_parts(segment: &str) -> impl Iterator<Item = String> + '_ {
    let mut out = Vec::new();
    let mut chars = segment.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' {
            let mut inner = String::new();
            for n in chars.by_ref() {
                if n == c {
                    break;
                }
                inner.push(n);
            }
            if !inner.is_empty() {
                out.push(inner);
            }
        }
    }
    out.into_iter()
}

fn tokenize(segment: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in segment.chars() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else {
                cur.push(c);
            }
            continue;
        }
        match c {
            '\'' | '"' => quote = Some(c),
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn basename(cmd: &str) -> &str {
    cmd.rsplit(['/', '\\']).next().unwrap_or(cmd)
}

fn ssh_remote_command(args: &[String]) -> String {
    let mut i = 0;
    while i < args.len() {
        let t = &args[i];
        if t == "--" {
            return args[i + 1..].join(" ");
        }
        if t == "-p" || t == "-i" || t == "-l" || t == "-o" || t == "-F" || t == "-J" || t == "-W" {
            i += 2;
            continue;
        }
        if t.starts_with('-') {
            i += 1;
            continue;
        }
        return args[i + 1..].join(" ");
    }
    String::new()
}

fn first_command(segment: &str) -> Option<String> {
    let tokens = tokenize(segment);
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if t.contains('=') && !t.starts_with('-') && !t.starts_with('/') {
            i += 1;
            continue;
        }
        if t == "sudo" || t == "command" || t == "env" || t == "nice" || t == "nohup" {
            i += 1;
            continue;
        }
        return Some(basename(t).to_string());
    }
    None
}

fn git_subcommand(segment: &str) -> Option<String> {
    let tokens = tokenize(segment);
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if t.contains('=') && !t.starts_with('-') && !t.starts_with('/') {
            i += 1;
            continue;
        }
        if t == "sudo" || t == "command" || t == "env" || t == "nice" || t == "nohup" {
            i += 1;
            continue;
        }
        if t == "ssh" || basename(t) == "ssh" {
            return git_subcommand(&ssh_remote_command(&tokens[i + 1..]));
        }
        break;
    }
    if i >= tokens.len() || basename(&tokens[i]) != "git" {
        return None;
    }
    i += 1;
    while i < tokens.len() {
        let t = &tokens[i];
        if t == "-C" || t == "-c" || t == "--git-dir" || t == "--work-tree" {
            i += 2;
            continue;
        }
        if t.starts_with("--git-dir=") || t.starts_with("--work-tree=") {
            i += 1;
            continue;
        }
        if t.starts_with('-') {
            i += 1;
            continue;
        }
        return Some(t.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{blocked_kind, deny_cloud_github_cli};

    #[test]
    fn blocks_git_push_and_wrappers() {
        assert_eq!(blocked_kind("git push origin HEAD"), Some("git push"));
        assert_eq!(blocked_kind("git -C /tmp push --force"), Some("git push"));
        assert_eq!(blocked_kind("sudo git push"), Some("git push"));
        assert_eq!(
            blocked_kind("ssh host git push origin main"),
            Some("git push")
        );
        assert_eq!(
            blocked_kind("ssh host 'cd /src && git push'"),
            Some("git push")
        );
        assert!(deny_cloud_github_cli("git push")
            .unwrap()
            .contains("push or publish is restricted"));
    }

    #[test]
    fn blocks_gh() {
        assert_eq!(blocked_kind("gh pr create --fill"), Some("gh"));
        assert_eq!(blocked_kind("gh auth status"), Some("gh"));
        assert_eq!(blocked_kind("/usr/bin/gh repo view"), Some("gh"));
    }

    #[test]
    fn allows_local_git() {
        assert_eq!(blocked_kind("git status"), None);
        assert_eq!(blocked_kind("git diff --stat"), None);
        assert_eq!(blocked_kind("git add -A && git commit -m 'wip'"), None);
        assert_eq!(blocked_kind("git log --grep=push"), None);
        assert_eq!(blocked_kind("echo git push"), None);
    }
}
