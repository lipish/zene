use tempfile::tempdir;
use zene_core::{Agent, ZeneConfig};

#[tokio::test]
async fn test_agent_minimal_facade() {
    let dir = tempdir().unwrap();
    let config = ZeneConfig {
        provider: "anthropic".into(),
        anthropic_api_key: Some("test-key".into()),
        ..Default::default()
    };

    let agent = Agent::minimal(dir.path())
        .config(config)
        .bypass_permissions()
        .build()
        .await
        .expect("build minimal agent");

    let tools = agent.active_tool_names();
    assert!(tools.contains(&"Read".to_string()));
    assert!(tools.contains(&"Grep".to_string()));
    assert!(tools.contains(&"Glob".to_string()));
    assert!(!tools.contains(&"Bash".to_string()));
    assert!(!tools.contains(&"Write".to_string()));
    assert!(!tools.contains(&"Edit".to_string()));
}

#[tokio::test]
async fn test_agent_core_facade() {
    let dir = tempdir().unwrap();
    let config = ZeneConfig {
        provider: "anthropic".into(),
        anthropic_api_key: Some("test-key".into()),
        ..Default::default()
    };

    let agent = Agent::core(dir.path())
        .config(config)
        .bypass_permissions()
        .build()
        .await
        .expect("build core agent");

    let tools = agent.active_tool_names();
    assert!(tools.contains(&"Read".to_string()));
    assert!(tools.contains(&"Grep".to_string()));
    assert!(tools.contains(&"Glob".to_string()));
    assert!(tools.contains(&"Bash".to_string()));
    assert!(tools.contains(&"Write".to_string()));
    assert!(tools.contains(&"Edit".to_string()));
    // Builtin extras should not be present in core
    assert!(!tools.contains(&"Task".to_string()));
    assert!(!tools.contains(&"TodoWrite".to_string()));
}

#[tokio::test]
async fn test_agent_builder_defaults() {
    let dir = tempdir().unwrap();
    let config = ZeneConfig {
        provider: "anthropic".into(),
        anthropic_api_key: Some("test-key".into()),
        ..Default::default()
    };

    let agent = Agent::builder(dir.path())
        .config(config)
        .without_mcp()
        .build()
        .await
        .expect("build default agent");

    assert_eq!(agent.current_session_mode(), "default");
    assert!(!agent.is_plan_mode_active());
}
