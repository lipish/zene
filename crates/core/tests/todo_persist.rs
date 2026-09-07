use std::env;
use std::sync::Arc;

use tempfile::tempdir;
use zene_sandbox::LocalSandbox;
use zene_session::{SessionRecord, TodoStatus};
use zene_tools::{default_builtin_tools, shared_todo_store_from, ToolContext};

#[tokio::test]
async fn todo_write_persists_across_session_reload() {
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _guard = LOCK.lock().await;
    let home = tempfile::tempdir().expect("tempdir");
    let prev = env::var("ZENE_HOME").ok();
    env::set_var("ZENE_HOME", home.path());

    let dir = tempdir().unwrap();
    let mut session = SessionRecord::new(dir.path());
    let store = shared_todo_store_from(session.todos.clone());
    let ctx = ToolContext {
        sandbox: Arc::new(LocalSandbox::new(dir.path())),
        cancel: None,
        subagent: None,
        permission: None,
        plan_mode: None,
        todos: Some(Arc::clone(&store)),
        ask_user: None,
        background: None,
    };

    let result = default_builtin_tools()
        .execute(
            "TodoWrite",
            r#"{
                "todos": [
                    { "id": "persist-1", "content": "Save todos", "status": "in_progress" },
                    { "id": "persist-2", "content": "Reload session", "status": "pending" }
                ]
            }"#,
            &ctx,
        )
        .await
        .expect("TodoWrite should run");
    assert!(!result.is_error);

    {
        let store = store.lock();
        session.todos = store.to_items();
    }

    session.save().expect("save session");
    let loaded = SessionRecord::load(&session.meta.id).expect("load session");
    assert_eq!(loaded.todos.len(), 2);
    assert_eq!(loaded.todos[0].id, "persist-1");
    assert_eq!(loaded.todos[0].content, "Save todos");
    assert_eq!(loaded.todos[0].status, TodoStatus::InProgress);
    assert_eq!(loaded.todos[1].id, "persist-2");
    assert_eq!(loaded.todos[1].content, "Reload session");
    assert_eq!(loaded.todos[1].status, TodoStatus::Pending);

    let reloaded_store = shared_todo_store_from(loaded.todos);
    let list_ctx = ToolContext {
        sandbox: Arc::new(LocalSandbox::new(dir.path())),
        cancel: None,
        subagent: None,
        permission: None,
        plan_mode: None,
        todos: Some(reloaded_store),
        ask_user: None,
        background: None,
    };
    let list_result = default_builtin_tools()
        .execute("TodoList", "{}", &list_ctx)
        .await
        .expect("TodoList should run");
    assert!(!list_result.is_error);
    assert!(list_result.content.contains("Save todos"));
    assert!(list_result.content.contains("Reload session"));

    match prev {
        Some(value) => env::set_var("ZENE_HOME", value),
        None => env::remove_var("ZENE_HOME"),
    }
}
