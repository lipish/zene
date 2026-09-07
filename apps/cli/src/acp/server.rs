use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};
use zene_agent_runtime::{ApprovalDecision, RuntimeHandle};
use zene_config::ZeneConfig;
use zene_core::{
    Agent, ApprovalRequest, AskUserOption, PermissionGate, PermissionMode, PromptChoice,
};
use zene_runtime::{RuntimeControl, RuntimeRecoveryInfo};
use zene_sandbox::LocalSandbox;
use zene_session::{list_sessions_for_workdir, SessionRecord};
use zene_turn::{RuntimeEvent, RuntimeEventKind};

use super::fs_bridge::AcpRemoteFs;
use super::protocol::{
    err_response, error_codes, is_notification, is_request, is_response, ok_response,
    prompt_text_from_params, RpcId,
};
use super::terminal_bridge::AcpRemoteTerminal;
use super::transport::{AcpWriter, SharedState};
use super::updates::{
    agent_message_chunk, agent_thought_chunk, available_commands_update, current_mode_update,
    error_update, lifecycle_event_update, modes_state, plan_from_todo_arguments,
    projection_ready_update, replay_updates_from_messages, step_started, tool_call_result_update,
    tool_call_update, tool_kind, tool_title, turn_ended, turn_started, usage_update,
};

/// Tracks the tool call currently awaiting permission so ACP can reuse its id.
#[derive(Default)]
struct PendingToolCall {
    id: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct ClientCapabilities {
    fs_read: bool,
    fs_write: bool,
    terminal: bool,
}

struct QueuedPrompt {
    rpc_id: RpcId,
    params: Value,
}

struct ActivePrompt {
    session_id: String,
    rpc_id: RpcId,
    result_rx: oneshot::Receiver<Result<Value>>,
}

struct AcpSession {
    runtime: Arc<dyn RuntimeControl>,
    busy: bool,
    /// Last tool call id observed via runtime events (used by permission prompts).
    pending_tool: Arc<Mutex<PendingToolCall>>,
    prompt_queue: VecDeque<QueuedPrompt>,
    permission_mode: String,
}

pub struct AcpServer {
    workdir: PathBuf,
    yolo: bool,
    sessions: HashMap<String, AcpSession>,
    writer: AcpWriter,
    client_caps: ClientCapabilities,
}

/// Run the ACP stdio agent until stdin closes.
pub async fn run_acp(workdir: PathBuf, yolo: bool) -> Result<()> {
    AcpServer::run(workdir, yolo).await
}

impl AcpServer {
    async fn run(workdir: PathBuf, yolo: bool) -> Result<()> {
        let shared = Arc::new(Mutex::new(SharedState::new()));
        let (out_tx, mut out_rx) = mpsc::unbounded_channel::<String>();
        let writer = AcpWriter {
            tx: out_tx,
            shared: Arc::clone(&shared),
        };

        let stdout_task = tokio::task::spawn_blocking(move || {
            let mut stdout = std::io::stdout().lock();
            while let Some(line) = out_rx.blocking_recv() {
                if writeln!(stdout, "{line}").is_err() {
                    break;
                }
                let _ = stdout.flush();
            }
        });

        let (in_tx, mut in_rx) = mpsc::unbounded_channel::<Value>();
        let shared_reader = Arc::clone(&shared);
        let stdin_task = tokio::task::spawn_blocking(move || {
            let stdin = std::io::stdin();
            let reader = BufReader::new(stdin.lock());
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(msg) => {
                        if is_response(&msg) {
                            let id = match &msg["id"] {
                                Value::Number(n) => n.to_string(),
                                Value::String(s) => s.clone(),
                                _ => continue,
                            };
                            let pending = {
                                let mut g = shared_reader.lock().unwrap();
                                g.take_pending(&id)
                            };
                            if let Some(tx) = pending {
                                if let Some(err) = msg.get("error") {
                                    let _ = tx.send(Err(err.clone()));
                                } else {
                                    let _ = tx.send(Ok(msg
                                        .get("result")
                                        .cloned()
                                        .unwrap_or(Value::Null)));
                                }
                            }
                        } else if in_tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("ACP: invalid JSON line: {e}");
                    }
                }
            }
        });

        let mut server = Self {
            workdir,
            yolo,
            sessions: HashMap::new(),
            writer,
            client_caps: ClientCapabilities::default(),
        };
        let mut active: Option<ActivePrompt> = None;
        let mut stdin_open = true;

        while stdin_open || active.is_some() {
            if let Some(mut current) = active.take() {
                tokio::select! {
                    msg = in_rx.recv(), if stdin_open => {
                        match msg {
                            Some(msg) => {
                                if let Err(e) = server
                                    .handle_incoming(msg, Some(&mut current), &mut active)
                                    .await
                                {
                                    warn!("ACP incoming while prompting: {e:#}");
                                }
                                if active.is_none() {
                                    active = Some(current);
                                }
                            }
                            None => {
                                stdin_open = false;
                                if let Some(sess) = server.sessions.get(&current.session_id) {
                                    let _ = sess.runtime.cancel().await;
                                }
                                active = Some(current);
                            }
                        }
                    }
                    result = &mut current.result_rx => {
                        let reply = match result {
                            Ok(Ok(value)) => ok_response(current.rpc_id.clone(), value),
                            Ok(Err(err)) => {
                                warn!("ACP session/prompt: {err:#}");
                                err_response(
                                    current.rpc_id.clone(),
                                    dispatch_error_code("session/prompt", &err),
                                    format!("{err:#}"),
                                )
                            }
                            Err(_) => err_response(
                                current.rpc_id.clone(),
                                error_codes::APPLICATION_ERROR,
                                "prompt worker dropped",
                            ),
                        };
                        if let Err(e) = server.writer.send_raw(reply.to_string()) {
                            warn!("ACP write failed: {e}");
                            break;
                        }
                        if let Some(sess) = server.sessions.get_mut(&current.session_id) {
                            sess.busy = false;
                            sess.pending_tool.lock().unwrap().id = None;
                        }
                        if let Err(e) = server.maybe_start_queued_prompt(&current.session_id, &mut active).await {
                            warn!("ACP queued prompt: {e:#}");
                        }
                    }
                }
            } else {
                match in_rx.recv().await {
                    Some(msg) => {
                        if let Err(e) = server.handle_incoming(msg, None, &mut active).await {
                            warn!("ACP incoming: {e:#}");
                        }
                    }
                    None => {
                        stdin_open = false;
                    }
                }
            }
        }

        for (_, sess) in server.sessions.drain() {
            let _ = sess.runtime.shutdown().await;
        }
        drop(server.writer);
        let _ = stdin_task.await;
        let _ = stdout_task.await;
        Ok(())
    }

    async fn handle_incoming(
        &mut self,
        msg: Value,
        current: Option<&mut ActivePrompt>,
        active: &mut Option<ActivePrompt>,
    ) -> Result<()> {
        if is_notification(&msg) {
            let method = msg["method"].as_str().unwrap_or("");
            if method == "session/cancel" {
                if let Some(sid) = msg["params"]["sessionId"].as_str() {
                    if let Some(s) = self.sessions.get(sid) {
                        let _ = s.runtime.cancel().await;
                    }
                    if let Some(s) = self.sessions.get_mut(sid) {
                        s.prompt_queue.clear();
                    }
                    if let Some(cur) = current {
                        if cur.session_id == sid {
                            if let Some(s) = self.sessions.get(sid) {
                                let _ = s.runtime.cancel().await;
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        if !is_request(&msg) {
            return Ok(());
        }
        let id = RpcId::from_value(&msg["id"]);
        let method = msg["method"].as_str().unwrap_or("").to_string();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        if method == "session/prompt" {
            if let Err(e) = self
                .enqueue_or_start_prompt(id.clone(), params, active)
                .await
            {
                warn!("ACP {method}: {e:#}");
                let reply = err_response(id, dispatch_error_code(&method, &e), format!("{e:#}"));
                self.writer.send_raw(reply.to_string())?;
            }
            Ok(())
        } else {
            let reply = match self.dispatch(&method, params).await {
                Ok(result) => ok_response(id, result),
                Err(e) => {
                    warn!("ACP {method}: {e:#}");
                    err_response(id, dispatch_error_code(&method, &e), format!("{e:#}"))
                }
            };
            self.writer.send_raw(reply.to_string())?;
            Ok(())
        }
    }

    async fn dispatch(&mut self, method: &str, params: Value) -> Result<Value> {
        match method {
            "initialize" => self.handle_initialize(params),
            "session/new" => self.handle_session_new(params).await,
            "session/load" => self.handle_session_load(params).await,
            "session/resume" => self.handle_session_resume(params).await,
            "session/list" => self.handle_session_list(params),
            "session/close" => self.handle_session_close(params).await,
            "session/set_mode" => self.handle_session_set_mode(params).await,
            "session/activate_tools" => self.handle_session_activate_tools(params).await,
            "session/deactivate_tools" => self.handle_session_deactivate_tools(params).await,
            "session/set_config_option" => self.handle_session_set_config_option(params).await,
            "session/clear_queue" => self.handle_session_clear_queue(params),
            "session/steer" => self.handle_session_steer(params).await,
            "session/follow_up" => self.handle_session_follow_up(params).await,
            "session/set_steering_mode" => self.handle_set_queue_mode(params, true).await,
            "session/set_follow_up_mode" => self.handle_set_queue_mode(params, false).await,
            "authenticate" => Ok(json!({})),
            "session/prompt" => Err(anyhow!(
                "session/prompt is handled by the async prompt queue"
            )),
            other => Err(MethodNotFound(other.to_string()).into()),
        }
    }

    fn handle_initialize(&mut self, params: Value) -> Result<Value> {
        let client_version = params
            .get("protocolVersion")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        if client_version != 1 {
            bail!("unsupported protocolVersion {client_version}; zene acp speaks 1");
        }

        let fs = params.pointer("/clientCapabilities/fs");
        self.client_caps = ClientCapabilities {
            fs_read: fs
                .and_then(|v| v.get("readTextFile"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            fs_write: fs
                .and_then(|v| v.get("writeTextFile"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            terminal: params
                .pointer("/clientCapabilities/terminal")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        if self.client_caps.fs_read || self.client_caps.fs_write || self.client_caps.terminal {
            debug!(
                fs_read = self.client_caps.fs_read,
                fs_write = self.client_caps.fs_write,
                terminal = self.client_caps.terminal,
                "ACP client advertised capabilities"
            );
        }

        Ok(json!({
            "protocolVersion": 1,
            "agentCapabilities": {
                "loadSession": true,
                "promptCapabilities": {
                    "image": false,
                    "audio": false,
                    "embeddedContext": true
                },
                "mcpCapabilities": {
                    "http": false,
                    "sse": false
                },
                "sessionCapabilities": {
                    "list": {},
                    "resume": {}
                }
            },
            "agentInfo": {
                "name": "zene",
                "title": "Zene",
                "version": env!("CARGO_PKG_VERSION")
            },
            "authMethods": []
        }))
    }

    async fn handle_session_new(&mut self, params: Value) -> Result<Value> {
        let cwd = resolve_cwd(&params, &self.workdir)?;
        let session = SessionRecord::new(&cwd);
        let id = session.meta.id.clone();
        let acp_session = self.build_session(session, &cwd, &id, false).await?;
        let mode = acp_session.runtime.current_mode().await?;
        let response = with_recovery_metadata(
            json!({
                "sessionId": id,
                "modes": modes_state(&mode),
            }),
            acp_session.runtime.as_ref(),
        )?;
        self.sessions.insert(id.clone(), acp_session);
        self.advertise_session(&id)?;
        Ok(response)
    }

    async fn handle_session_load(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let cwd = resolve_cwd(&params, &self.workdir)?;
        let session = SessionRecord::repair_legacy(&sid).context("load session")?;
        let updates = replay_updates_from_messages(&session.view().messages);
        let acp_session = self.build_session(session, &cwd, &sid, false).await?;
        let mode = acp_session.runtime.current_mode().await?;
        let response = with_recovery_metadata(
            json!({
                "sessionId": sid,
                "modes": modes_state(&mode),
            }),
            acp_session.runtime.as_ref(),
        )?;
        self.sessions.insert(sid.clone(), acp_session);

        // ACP requires replaying history via session/update before responding.
        for update in updates {
            let mut update = update;
            if let Some(obj) = update.as_object_mut() {
                obj.insert("_meta".into(), json!({ "isReplay": true }));
            }
            self.writer.session_update(&sid, update)?;
        }
        self.advertise_session(&sid)?;

        Ok(response)
    }

    async fn handle_session_resume(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let cwd = resolve_cwd(&params, &self.workdir)?;
        let session = SessionRecord::repair_legacy(&sid).context("resume session")?;
        // Resume restores context without replaying history.
        let acp_session = self.build_session(session, &cwd, &sid, true).await?;
        let mode = acp_session.runtime.current_mode().await?;
        let response = with_recovery_metadata(
            json!({
                "sessionId": sid,
                "modes": modes_state(&mode),
            }),
            acp_session.runtime.as_ref(),
        )?;
        self.sessions.insert(sid.clone(), acp_session);
        self.advertise_session(&sid)?;
        Ok(response)
    }

    fn handle_session_list(&self, params: Value) -> Result<Value> {
        let cwd = params
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| self.workdir.clone());
        let cwd = cwd.canonicalize().unwrap_or(cwd);
        let sessions = list_sessions_for_workdir(&cwd).context("list sessions")?;
        let sessions: Vec<Value> = sessions
            .into_iter()
            .map(|meta| {
                json!({
                    "sessionId": meta.id,
                    "cwd": meta.workdir,
                    "title": meta.title,
                    "updatedAt": meta.updated_at.to_rfc3339(),
                })
            })
            .collect();
        Ok(json!({ "sessions": sessions }))
    }

    async fn handle_session_close(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let Some(sess) = self.sessions.remove(&sid) else {
            bail!("unknown sessionId: {sid}");
        };
        let _ = sess.runtime.shutdown().await;
        Ok(json!({}))
    }

    async fn handle_session_set_mode(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let mode_id = params
            .get("modeId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("modeId required"))?
            .to_string();
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        let active = sess.runtime.set_mode(mode_id).await?;
        self.writer
            .session_update(&sid, current_mode_update(&active))?;
        Ok(json!({}))
    }

    async fn handle_session_activate_tools(&mut self, params: Value) -> Result<Value> {
        self.handle_session_tool_activation(params, true).await
    }

    async fn handle_session_deactivate_tools(&mut self, params: Value) -> Result<Value> {
        self.handle_session_tool_activation(params, false).await
    }

    async fn handle_session_tool_activation(
        &mut self,
        params: Value,
        activate: bool,
    ) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let names = params
            .get("toolNames")
            .or_else(|| params.get("tools"))
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("toolNames array required"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("toolNames must contain strings"))
            })
            .collect::<Result<Vec<_>>>()?;
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        let changed = if activate {
            sess.runtime.activate_tools(names).await?
        } else {
            sess.runtime.deactivate_tools(names).await?
        };
        Ok(json!({ "toolNames": changed }))
    }

    async fn handle_session_set_config_option(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let key = params
            .get("key")
            .or_else(|| params.get("name"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("config key or name required"))?;
        let value = params
            .get("value")
            .ok_or_else(|| anyhow!("config value required"))?;

        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;

        match key {
            "mode" => {
                let mode_id = value
                    .as_str()
                    .ok_or_else(|| anyhow!("mode must be a string"))?;
                let active = sess.runtime.set_mode(mode_id.to_string()).await?;
                self.writer
                    .session_update(&sid, current_mode_update(&active))?;
            }
            "permission_mode" | "permissionMode" => {
                let mode_str = value
                    .as_str()
                    .ok_or_else(|| anyhow!("permission_mode must be a string"))?;
                sess.permission_mode = mode_str.to_string();
            }
            _ => {
                debug!(session = %sid, key = %key, value = %value, "ACP runtime config option updated");
            }
        }

        Ok(json!({
            "key": key,
            "value": value,
        }))
    }

    fn handle_session_clear_queue(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        let cleared_count = sess.prompt_queue.len();
        let mut cleared = Vec::new();
        while let Some(q) = sess.prompt_queue.pop_front() {
            cleared.push(q.rpc_id.to_value());
        }
        debug!(session = %sid, cleared_count, "ACP prompt queue cleared");
        Ok(json!({
            "clearedCount": cleared_count,
            "cleared": cleared,
            "steering": 0,
            "followUp": 0,
            "promptQueue": cleared_count,
        }))
    }

    async fn handle_session_steer(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let text = params
            .get("text")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| prompt_text_from_params(&params));
        if text.trim().is_empty() {
            bail!("text required");
        }
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        sess.runtime.steer(text).await?;
        Ok(json!({}))
    }

    async fn handle_session_follow_up(&mut self, params: Value) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let text = params
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| prompt_text_from_params(&params));
        if text.trim().is_empty() {
            bail!("text required");
        }
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        sess.runtime.follow_up(text).await?;
        Ok(json!({}))
    }

    async fn handle_set_queue_mode(&mut self, params: Value, steering: bool) -> Result<Value> {
        let sid = params
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let mode = match params.get("mode").and_then(Value::as_str) {
            Some("one-at-a-time") => zene_turn::QueueMode::OneAtATime,
            Some("all") => zene_turn::QueueMode::All,
            _ => bail!("mode must be one-at-a-time or all"),
        };
        let sess = self
            .sessions
            .get(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        if steering {
            sess.runtime.set_steering_mode(mode).await?;
        } else {
            sess.runtime.set_follow_up_mode(mode).await?;
        }
        Ok(json!({ "mode": params["mode"] }))
    }

    fn advertise_session(&self, session_id: &str) -> Result<()> {
        self.writer
            .session_update(session_id, available_commands_update())?;
        Ok(())
    }

    async fn build_session(
        &self,
        session: SessionRecord,
        cwd: &Path,
        session_id: &str,
        automatic_recovery: bool,
    ) -> Result<AcpSession> {
        let config = ZeneConfig::load(cwd).map_err(|err| anyhow!(err.to_string()))?;
        let permission_mode = if self.yolo {
            PermissionMode::BypassPermissions
        } else {
            PermissionMode::parse(&config.permission_mode)
        };
        let mut sandbox = LocalSandbox::with_keel(cwd)
            .await
            .context("initialize Keel execution layer")?;
        if self.client_caps.fs_read || self.client_caps.fs_write {
            sandbox = sandbox.with_remote_fs(Arc::new(AcpRemoteFs::new(
                self.writer.clone(),
                session_id,
                self.client_caps.fs_read,
                self.client_caps.fs_write,
            )));
        }
        if self.client_caps.terminal {
            sandbox = sandbox.with_remote_terminal(Arc::new(AcpRemoteTerminal::new(
                self.writer.clone(),
                session_id,
            )));
        }
        let mut agent = Agent::new(config, sandbox, session, permission_mode).await?;
        let pending_tool = Arc::new(Mutex::new(PendingToolCall::default()));
        configure_agent_brokers(
            &mut agent,
            self.writer.clone(),
            session_id.to_string(),
            self.yolo,
            permission_mode,
        );
        let (runtime, _task) = if automatic_recovery {
            RuntimeHandle::spawn_with_automatic_recovery(agent)
        } else {
            RuntimeHandle::spawn(agent)
        };
        Ok(AcpSession {
            runtime: Arc::new(runtime),
            busy: false,
            pending_tool,
            prompt_queue: VecDeque::new(),
            permission_mode: permission_mode.as_str().to_string(),
        })
    }

    async fn enqueue_or_start_prompt(
        &mut self,
        rpc_id: RpcId,
        params: Value,
        active: &mut Option<ActivePrompt>,
    ) -> Result<()> {
        let sid = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("sessionId required"))?
            .to_string();
        let text = prompt_text_from_params(&params);
        if text.trim().is_empty() {
            bail!("empty prompt");
        }
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;

        if sess.busy {
            let behavior = params
                .get("streamingBehavior")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("streamingBehavior is required when session is busy"))?;
            match behavior {
                "steer" => {
                    sess.runtime.steer(text).await?;
                    self.writer
                        .send_raw(ok_response(rpc_id, json!({})).to_string())?;
                    return Ok(());
                }
                "followUp" => {
                    sess.runtime.follow_up(text).await?;
                    self.writer
                        .send_raw(ok_response(rpc_id, json!({})).to_string())?;
                    return Ok(());
                }
                other => bail!("unsupported streamingBehavior: {other}"),
            }
        }

        self.start_prompt(sid, rpc_id, text, active).await
    }

    async fn maybe_start_queued_prompt(
        &mut self,
        session_id: &str,
        active: &mut Option<ActivePrompt>,
    ) -> Result<()> {
        loop {
            let next = self
                .sessions
                .get_mut(session_id)
                .and_then(|sess| sess.prompt_queue.pop_front());
            let Some(next) = next else {
                return Ok(());
            };
            let text = prompt_text_from_params(&next.params);
            if text.trim().is_empty() {
                let reply = err_response(next.rpc_id, error_codes::INVALID_PARAMS, "empty prompt");
                self.writer.send_raw(reply.to_string())?;
                continue;
            }
            return self
                .start_prompt(session_id.to_string(), next.rpc_id, text, active)
                .await;
        }
    }

    async fn start_prompt(
        &mut self,
        sid: String,
        rpc_id: RpcId,
        text: String,
        active: &mut Option<ActivePrompt>,
    ) -> Result<()> {
        let writer = self.writer.clone();
        let sess = self
            .sessions
            .get_mut(&sid)
            .ok_or_else(|| anyhow!("unknown sessionId: {sid}"))?;
        sess.busy = true;
        let runtime = sess.runtime.clone();
        let pending_tool = Arc::clone(&sess.pending_tool);
        let permission_mode = sess.permission_mode.clone();

        let (tx, rx) = oneshot::channel();
        *active = Some(ActivePrompt {
            session_id: sid.clone(),
            rpc_id,
            result_rx: rx,
        });

        tokio::spawn(async move {
            let result =
                run_prompt_job(runtime, writer, sid, text, pending_tool, permission_mode).await;
            let _ = tx.send(result);
        });
        Ok(())
    }
}

fn configure_agent_brokers(
    agent: &mut Agent,
    writer: AcpWriter,
    sid: String,
    yolo: bool,
    permission_mode: PermissionMode,
) {
    if !yolo {
        agent.set_permission_gate(PermissionGate::new(permission_mode));
        agent.enable_runtime_approval_waiters();
    }

    let writer_ask = writer.clone();
    agent.set_ask_user_prompter(Arc::new(move |question, options| {
        acp_ask_user_prompt(&writer_ask, &sid, question, options)
    }));
}

async fn run_prompt_job(
    runtime: Arc<dyn RuntimeControl>,
    writer: AcpWriter,
    sid: String,
    text: String,
    pending_tool: Arc<Mutex<PendingToolCall>>,
    permission_mode: String,
) -> Result<Value> {
    let mut events = runtime.subscribe();
    let prompt_id = uuid::Uuid::new_v4().to_string();
    let result = runtime.prompt(text);
    tokio::pin!(result);

    loop {
        tokio::select! {
            response = &mut result => {
                match response {
                    Ok(_) => {
                        debug!(session = %sid, "ACP prompt completed");
                        return Ok(json!({ "stopReason": "end_turn" }));
                    }
                    Err(err) if err.to_string().contains("cancel") || err.to_string().contains("aborted") => {
                        return Ok(json!({ "stopReason": "cancelled" }));
                    }
                    Err(err) => return Err(err),
                }
            }
            event = events.recv() => {
                match event {
                    Ok(event) => {
                        if let RuntimeEventKind::ApprovalRequested {
                            request_id,
                            tool_name,
                            arguments,
                            tool_call_id,
                        } = &event.kind
                        {
                            tokio::spawn(handle_runtime_approval(RuntimeApprovalParams {
                                runtime: Arc::clone(&runtime),
                                writer: writer.clone(),
                                session_id: sid.clone(),
                                permission_mode: permission_mode.clone(),
                                pending_tool: Arc::clone(&pending_tool),
                                request_id: request_id.clone(),
                                tool_name: tool_name.clone(),
                                arguments: arguments.clone(),
                                tool_call_id: tool_call_id.clone(),
                            }));
                        }
                        project_runtime_event(
                            &writer,
                            &event,
                            &sid,
                            &prompt_id,
                            &pending_tool,
                        );
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        return Err(anyhow!("runtime event stream closed"));
                    }
                }
            }
        }
    }
}

struct RuntimeApprovalParams {
    runtime: Arc<dyn RuntimeControl>,
    writer: AcpWriter,
    session_id: String,
    permission_mode: String,
    pending_tool: Arc<Mutex<PendingToolCall>>,
    request_id: String,
    tool_name: String,
    arguments: String,
    tool_call_id: Option<String>,
}

async fn handle_runtime_approval(params: RuntimeApprovalParams) {
    let tool_call_id = params.tool_call_id.unwrap_or_else(|| {
        params
            .pending_tool
            .lock()
            .unwrap()
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
    });
    let preview = ApprovalRequest {
        request_id: params.request_id.clone(),
        tool_name: params.tool_name.clone(),
        arguments: params.arguments,
        tool_call_id: Some(tool_call_id.clone()),
    }
    .preview();
    let decision = match acp_permission_prompt(
        &params.writer,
        &params.session_id,
        &params.permission_mode,
        &params.tool_name,
        &preview,
        &tool_call_id,
    )
    .await
    {
        Ok(choice) => approval_decision(choice),
        Err(err) => {
            warn!(error = %err, request_id = %params.request_id, "ACP permission prompt failed");
            ApprovalDecision::Deny
        }
    };
    if let Err(err) = params.runtime.approve(params.request_id, decision).await {
        warn!(error = %err, "runtime approval resolve failed");
    }
}

fn approval_decision(choice: PromptChoice) -> ApprovalDecision {
    match choice {
        PromptChoice::AllowOnce => ApprovalDecision::AllowOnce,
        PromptChoice::AllowSession => ApprovalDecision::AllowSession,
        PromptChoice::Deny => ApprovalDecision::Deny,
    }
}

fn project_runtime_event(
    writer: &AcpWriter,
    event: &RuntimeEvent,
    sid: &str,
    prompt_id: &str,
    pending_tool: &Arc<Mutex<PendingToolCall>>,
) {
    let event_id = format!("{prompt_id}-{}", event.sequence.value());
    let mut meta = json!({
        "promptId": prompt_id,
        "eventId": event_id,
        "sequence": event.sequence.value(),
        "isReplay": false,
    });
    let update = match &event.kind {
        RuntimeEventKind::TextDelta { delta } => Some(agent_message_chunk(delta)),
        RuntimeEventKind::ThoughtDelta { delta } => Some(agent_thought_chunk(delta)),
        RuntimeEventKind::ToolCall {
            id,
            name,
            arguments,
        } => {
            pending_tool.lock().unwrap().id = Some(id.to_string());
            let update = tool_call_update(id.as_str(), name, arguments, "pending");
            if name == "TodoWrite" {
                if let Some(mut plan) = plan_from_todo_arguments(arguments) {
                    attach_meta(
                        &mut plan,
                        json!({ "promptId": prompt_id, "isReplay": false }),
                    );
                    let _ = writer.session_update(sid, plan);
                }
            }
            Some(update)
        }
        RuntimeEventKind::ToolResult {
            id,
            content,
            is_error,
            duration_ms,
            ..
        } => {
            pending_tool.lock().unwrap().id = None;
            if let Some(ms) = duration_ms {
                meta["durationMs"] = json!(ms);
            }
            Some(tool_call_result_update(id.as_str(), content, *is_error))
        }
        RuntimeEventKind::StateChanged { state } => Some(current_mode_update(state)),
        RuntimeEventKind::UsageUpdate {
            usage,
            context_tokens,
            context_window,
            context_percent,
            context_epoch,
        } => Some(usage_update(
            u64::from(*context_tokens),
            u64::from((*context_window).max(1)),
            usage.prompt_tokens,
            usage.completion_tokens,
            *context_percent,
            usage.cached_tokens,
            *context_epoch,
        )),
        RuntimeEventKind::Error { message } => Some(error_update(message)),
        RuntimeEventKind::ProjectionReady(ready) => Some(projection_ready_update(ready)),
        RuntimeEventKind::TurnStarted => {
            let turn_id = event.turn_id.as_ref().map(|id| id.to_string());
            Some(turn_started(turn_id.as_deref()))
        }
        RuntimeEventKind::StepStarted { step } => {
            let turn_id = event.turn_id.as_ref().map(|id| id.to_string());
            Some(step_started(*step, turn_id.as_deref()))
        }
        RuntimeEventKind::TurnEnded { steps } => {
            let turn_id = event.turn_id.as_ref().map(|id| id.to_string());
            Some(turn_ended(*steps, turn_id.as_deref()))
        }
        RuntimeEventKind::LifecycleEvent { event, payload } => {
            Some(lifecycle_event_update(event, payload))
        }
        RuntimeEventKind::SteerInput { .. }
        | RuntimeEventKind::ApprovalRequested { .. }
        | RuntimeEventKind::ApprovalResolved { .. } => None,
    };
    if let Some(mut update) = update {
        attach_meta(&mut update, meta);
        let _ = writer.session_update(sid, update);
    }
}

fn attach_meta(update: &mut Value, meta: Value) {
    if let Some(obj) = update.as_object_mut() {
        // usage_update already has _meta; merge prompt metadata into it.
        if let Some(existing) = obj.get_mut("_meta").and_then(|v| v.as_object_mut()) {
            if let Some(extra) = meta.as_object() {
                for (k, v) in extra {
                    existing.insert(k.clone(), v.clone());
                }
            }
        } else {
            obj.insert("_meta".into(), meta);
        }
    }
}

fn with_recovery_metadata(mut response: Value, runtime: &dyn RuntimeControl) -> Result<Value> {
    let info = runtime.recovery_info()?;
    if let Some(obj) = response.as_object_mut() {
        obj.insert(
            "_meta".into(),
            json!({
                "recovery": recovery_info_metadata(&info),
            }),
        );
    }
    Ok(response)
}

fn recovery_info_metadata(info: &RuntimeRecoveryInfo) -> Value {
    json!({
        "disposition": info.disposition,
        "hasIncompleteExecution": info.has_incomplete_execution,
        "activeTurnCount": info.active_turn_count,
        "activeToolCount": info.active_tool_count,
        "safeResumeAllowed": info.safe_resume_allowed,
        "automaticResume": info.automatic_resume,
        "reason": info.reason,
    })
}

fn resolve_cwd(params: &Value, fallback: &Path) -> Result<PathBuf> {
    let cwd = params
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(|| fallback.to_path_buf());
    cwd.canonicalize()
        .with_context(|| format!("invalid cwd: {}", cwd.display()))
}

fn acp_ask_user_prompt(
    writer: &AcpWriter,
    session_id: &str,
    question: &str,
    options: Option<&[AskUserOption]>,
) -> io::Result<String> {
    let (tx, rx) = std::sync::mpsc::channel::<Result<Value, String>>();
    let writer = writer.clone();
    let session_id = session_id.to_string();
    let question = question.to_string();
    let option_labels: Vec<(String, String, Option<String>)> = options
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .map(|(idx, opt)| {
            (
                format!("ask-{idx}"),
                opt.label.clone(),
                opt.description.clone(),
            )
        })
        .collect();
    let option_labels_for_req = option_labels.clone();
    let handle = tokio::runtime::Handle::current();
    handle.spawn(async move {
        let mut perm_options = Vec::new();
        for (option_id, name, description) in &option_labels_for_req {
            let mut entry = json!({
                "optionId": option_id,
                "name": name,
                "kind": "allow_once",
            });
            if let Some(desc) = description {
                entry
                    .as_object_mut()
                    .unwrap()
                    .insert("description".into(), json!(desc));
            }
            perm_options.push(entry);
        }
        perm_options.push(json!({
            "optionId": "free-text",
            "name": "Type an answer",
            "kind": "allow_once",
        }));
        let raw_options: Vec<Value> = option_labels_for_req
            .iter()
            .map(|(_, label, description)| {
                json!({
                    "label": label,
                    "description": description,
                })
            })
            .collect();
        let result = writer
            .request(
                "session/request_permission",
                json!({
                    "sessionId": session_id,
                    "toolCall": {
                        "toolCallId": format!("ask_{}", uuid::Uuid::new_v4().simple()),
                        "title": question.clone(),
                        "kind": "other",
                        "status": "pending",
                        "rawInput": {
                            "askUser": true,
                            "question": question,
                            "options": raw_options,
                        },
                    },
                    "options": perm_options,
                }),
            )
            .await
            .map_err(|e| e.to_string());
        let _ = tx.send(result);
    });

    let result = std::thread::spawn(move || rx.recv())
        .join()
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "ask-user thread panicked"))?
        .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e))?
        .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e))?;

    let option_id = result
        .pointer("/outcome/optionId")
        .or_else(|| result.get("optionId"))
        .and_then(|v| v.as_str())
        .unwrap_or("free-text");

    if option_id == "reject-once" || option_id == "deny" {
        return Ok(String::new());
    }

    if option_id == "free-text" {
        let answer = result
            .pointer("/outcome/answer")
            .or_else(|| result.get("answer"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return Ok(answer);
    }

    if let Some((_, label, _)) = option_labels.iter().find(|(id, _, _)| id == option_id) {
        return Ok(label.clone());
    }

    // Client may echo the chosen label as optionId.
    Ok(option_id.to_string())
}

async fn acp_permission_prompt(
    writer: &AcpWriter,
    session_id: &str,
    permission_mode: &str,
    tool_name: &str,
    preview: &str,
    tool_call_id: &str,
) -> io::Result<PromptChoice> {
    let raw_input =
        serde_json::from_str::<Value>(preview).unwrap_or_else(|_| json!({ "preview": preview }));
    let result = writer
        .request(
            "session/request_permission",
            json!({
                "sessionId": session_id,
                "toolCall": {
                    "toolCallId": tool_call_id,
                    "title": tool_title(tool_name, preview),
                    "kind": tool_kind(tool_name),
                    "status": "pending",
                    "rawInput": raw_input,
                },
                "options": [
                    {
                        "optionId": "allow-once",
                        "name": "Allow once",
                        "kind": "allow_once"
                    },
                    {
                        "optionId": "allow-always",
                        "name": "Allow always",
                        "kind": "allow_always"
                    },
                    {
                        "optionId": "reject-once",
                        "name": "Reject",
                        "kind": "reject_once"
                    }
                ],
                "permissionMode": permission_mode,
            }),
        )
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))?;

    let option_id = result
        .pointer("/outcome/optionId")
        .or_else(|| result.get("optionId"))
        .and_then(|v| v.as_str())
        .unwrap_or("reject-once");

    Ok(match option_id {
        "allow-always" => PromptChoice::AllowSession,
        "allow-once" => PromptChoice::AllowOnce,
        _ => PromptChoice::Deny,
    })
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use zene_runtime::RuntimeRecoveryInfo;

    #[test]
    fn recovery_metadata_declares_no_automatic_resume() {
        let metadata = recovery_info_metadata(&RuntimeRecoveryInfo {
            disposition: "clean".into(),
            has_incomplete_execution: false,
            active_turn_count: 0,
            active_tool_count: 0,
            safe_resume_allowed: false,
            automatic_resume: false,
            reason: "no incomplete execution".into(),
        });
        assert_eq!(metadata["disposition"], "clean");
        assert_eq!(metadata["hasIncompleteExecution"], false);
        assert_eq!(metadata["automaticResume"], false);
    }

    #[test]
    fn approval_decision_maps_permission_choice() {
        assert_eq!(
            approval_decision(PromptChoice::AllowOnce),
            ApprovalDecision::AllowOnce
        );
        assert_eq!(
            approval_decision(PromptChoice::AllowSession),
            ApprovalDecision::AllowSession
        );
        assert_eq!(
            approval_decision(PromptChoice::Deny),
            ApprovalDecision::Deny
        );
    }
}

#[derive(Debug)]
struct MethodNotFound(String);

impl std::fmt::Display for MethodNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "method not found: {}", self.0)
    }
}

impl std::error::Error for MethodNotFound {}

fn dispatch_error_code(method: &str, err: &anyhow::Error) -> i64 {
    if err.downcast_ref::<MethodNotFound>().is_some()
        || err.to_string().starts_with("method not found:")
    {
        return error_codes::METHOD_NOT_FOUND;
    }
    let msg = err.to_string();
    if msg.contains("required")
        || msg.contains("empty prompt")
        || msg.contains("invalid cwd")
        || msg.contains("unsupported protocolVersion")
        || msg.contains("unknown sessionId")
        || msg.contains("unknown session mode")
        || msg.contains("already has an active prompt")
    {
        return error_codes::INVALID_PARAMS;
    }
    let _ = method;
    error_codes::APPLICATION_ERROR
}
