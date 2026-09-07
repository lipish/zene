# Changelog

## Unreleased

## v0.1.16 (2026-09-07)

### Added
- **Minimalist Agent Facade (`zene-core`)**:
  - Convenient factories `Agent::builder(workdir)`, `Agent::core(workdir)`, and `Agent::minimal(workdir)` with zero boilerplate setup.
  - Builder configuration helpers: `.core_tools()`, `.minimal_tools()`, `.bypass_permissions()`, `.config()`, `.session()`.
  - Re-exported core SDK types directly from `zene-core` (`ZeneConfig`, `ChatClient`, `LocalSandbox`, `Sandbox`, `SessionRecord`, `core_tools`, `minimal_tools`, `ToolRegistry`, `ToolCatalog`).
- **Core Toolsets Convergence (`zene-tools`)**:
  - `core_tools()`: lightweight harness toolset (`Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`) without external network or service coupling.
  - `minimal_tools()`: read-only inspection toolset (`Read`, `Grep`, `Glob`).
- **Pi-Aligned Turn Control & Extension Hooks**:
  - Support for steer and follow-up buffers (`SteerBuffer`, `FollowUpBuffer`).
  - Host-facing lifecycle hooks (`ExtensionHook`, `HookRunner`) and dynamic tool gating.

### Changed
- **Architectural Boundary Convergence**:
  - Decoupled cloud platform, web consoles, and multi-tenant control plane into `zene-cloud`.
  - Realigned `zene` as a dedicated, open, headless agent harness (ACP, turns, context projection, sandbox, durable session).
  - Merged `tool-runtime` into `zene-tools` and streamlined permission models.

## v0.1.15 (2026-08-23)

### Fixed
- Inference gateway: `publish_prefix` treats `409 CONFLICT` as an idempotent
  stale-epoch acknowledgement (info log) instead of a failure warning —
  mirrors `close_session` tolerance; the post-epoch-bump republish lands
  quietly (#129, Cortex field feedback P1).

### Docs
- Context architecture assessment (`docs/context-architecture-assessment.md`):
  A/B capability boundaries, gaps, and ordered optimization plan (#125),
  synced with v0.1.14 gateway changes.
- AGENTS.md: require PR flow for code and docs changes.

## v0.1.14 (2026-08-17)

### Changed
- Inference gateway integration hardening from Cortex field testing (issue #128):
  - Delta delivery now requires explicit `ZENE_CONTEXT_DELIVERY=delta` opt-in; gateway
    URL alone no longer enables it (no capability negotiation yet — a gateway that
    cannot rebuild full prompts would silently forward incomplete tails).
  - `publish_prefix` payload now carries `anchor_boundaries` (turn starts + tool-call
    group starts) so the gateway can score prefix liveness on harness-declared
    boundaries instead of tokenizer heuristics.
  - Gateway/ledger cache-hit tokens (`usage.cache_hit_tokens`, mirrors Cortex's
    `x-cortex-cache-hit-tokens`) flow through `TokenUsage.gateway_hit_tokens` into
    `PrefixCacheExplain`, ACP meta (`gatewayHitTokens`), and `/context` diagnostics,
    alongside provider `cached_tokens` for ledger-vs-engine drift diagnosis.
- deps: unigateway-sdk 2.14.0 → 2.14.2.

## v0.1.13 (2026-08-16)

### Added
- Context governance system inspired by DeepSeek Harness:
  - Agent Skills: `trim-cot-leakage`, `archive-agent-notes`, `find-simplifications` under `.agents/skills/`.
  - Cursor gate rule `.cursor/rules/trim-cot-leakage.mdc` to enforce HEAD-only prose and comments.
  - Agent Notes architecture and storage specification (`docs/agent-notes-design.md`).
  - Context optimization plan and governance lessons (`docs/context-optimization-plan.md`, `docs/deepseek-harness-context-lessons.md`).
- `zene-context`: `FsMemoryStore` automatically discovers and loads active Agent Notes (`.zene/notes/active/*.md` and `docs/notes/active/*.md`) into stable system prompt prefix.
- `zene-tools`: `OutputSanitizer` module to strip verbose passing test logs (`test ... ok`) and fold excessive command output (>300 lines) to save context tokens.
- Cloud Worker: Git Worktree multi-session isolation via `git worktree add --force --detach` from local `.repo-cache` with local clone fallback.
- Cloud Console UX:
  - Global keyboard shortcuts (`Cmd/Ctrl+B` toggle CodePanel, `Cmd/Ctrl+N` new task).
  - Prompt history navigation in empty Composer with `ArrowUp`/`ArrowDown`.
  - Diff reviewer checkboxes with automatic collapse and viewed state.
  - Large terminal log automatic folding (>30 lines / 2KB) with full output expand.
  - Dynamic tab title badges reflecting live agent state (`🟢`, `🟡`, `🔴`).
  - PromptQueue item cancellation while agent is running.
- Backend & DB: SQLite WAL journal mode, NORMAL synchronous, and 5s busy timeout tuning.

## v0.1.12 (2026-08-15)

### Added
- Cloud Console named capabilities: `import { … } from "@/cap/<id>"`, `./cloud/scripts/use-capability.sh`, and `./cloud/scripts/new-feature.sh` for a compiling UI+API slice.
- Shared Composer, typed `lib/cloud` clients, and API feature modules for LLM, repositories, and GitHub.

### Changed
- New Agent / Run follow-up pickers live in `components/ui` and `components/pickers`; AGENTS.md requires `@/cap/<id>` on Console generation.

## v0.1.11 (2026-08-11)

### Added
- `zene-context` crate: ContextEngine (estimate, compact, assemble, epoch) decoupled from runtime.
- `zene-inference-gateway`: UniGateway 2.14 session prefix store, delta assembly, optional Redis (`unigateway-session-redis`).
- Cloud deploy: inference gateway systemd unit, VM Redis in startup, worker `ZENE_INFERENCE_GATEWAY_URL` injection.
- Docs: `docs/context-engine.md`, agent-components cross-links; E2E test for publish + delta assembly.

### Changed
- OpenAI-compatible LLM path uses `unigateway-sdk` 2.14 with `_session_context` / fingerprint metadata.
- Context modules moved from `zene-core` to `zene-context`; Agent wires ContextEngine on prepare/usage.

## v0.1.10 (2026-07-26)

### Added
- Configurable run max turns (`ZENE_MAX_TURNS` / New Agent picker); `0` = unlimited, soft-stop instead of failing the run.
- Cloud worker supervisor scaling and run archive migration.
- Cloud Console markdown rendering and Cursor-style tool/activity summaries in Run view.

### Changed
- CLI focuses on `zene acp` (interactive REPL / headless `-p` removed).
- Opening a past run drains all event pages offline and commits the timeline once (no chunked replay).

### Fixed
- Run history reopen no longer appears to stream in slowly when events exceed the 500-per-page API limit.

## v0.1.8 (2026-07-25)

### Added
- Cloud Console: per-user BYOK LLM settings required before starting agents; worker injects credentials into `zene acp`.
- Cloud: reclaim stale worker leases and re-queue abandoned runs; ACP idle session hold for follow-ups.
- Cloud web design samples under `cloud/apps/web/sample/`.

### Changed
- Bump Keel sandbox stack to `eero-keel-core` 0.0.15 (baseline credential denies, audit hash chain, Windows Job/AppContainer). On Linux, Zene strips Keel FS deny rules before `Space::create` to avoid Keel 0.0.15’s outer-`bwrap` + Landlock `pre_exec` userns failure; host `path_policy` still blocks credential reads.
- Cloud Console UI: Changes / Diff / Git / Run / New Agent panels refined toward Cursor-style review workflow.
- `cloud/scripts/dev.sh` auto-builds or locates `zene` and prefers real ACP (mock only when allowed).

## v0.1.7 (2026-07-20)

Headless **Web Agent** becomes the default UI: local `zene-gateway` serves the browser UI over HTTP (long-polling + optional SSE), with `zene` / `zene web` as the launch entry. Releases and `www/install.sh` now ship both `zene` and `zene-gateway` binaries.

### Added
- Headless Web direction: `docs/WEB_AGENT_GATEWAY.md` design for HTTP Gateway + Web Agent UI (long-polling first; SSE optional; WebSocket not required).
- New `zene-gateway` binary (`apps/gateway`): thin local HTTP bridge over `zene acp` with token/Origin checks, `POST /api/v1/agents/{id}/messages`, cursor-based `GET /api/v1/agents/{id}/events` long polling, bootstrap/health, embedded Web Agent UI, and mock-ACP integration tests.
- Gateway phase B: optional SSE (`GET /events/stream`) with Web long-poll fallback, controller lease APIs, `apps/web-agent` UI (sessions/tool cards/usage/SSE), `--yolo`/`--sandbox-off`/`--acp-env`, and real `zene acp` + mock LLM smoke test.
- Gateway phase C: local ACP `terminal/*` host with Web terminal panel, Plan/Todo/background-task panels, mode switch + session close UI, and terminal roundtrip tests.
- Gateway phase D: on-disk event journal + agent meta, `restart`/`attach` recovery, poll backpressure and payload limits, `zene web` launcher, and `docs/GATEWAY_OPS.md`.
- Gateway phase E: AskUser over standard `session/request_permission`, Web `session/resume`, default `zene` launches Web Agent, remove ratatui TUI (`docs/TUI_MIGRATION.md`).
- Prefire two-pass compaction with `compaction_segments` persistence (NOTE₁ cache + sync merge).
- Memory flush / injection into context, with content-fingerprint dedup across turns.
- Intra **steps-first** pass: truncate current-turn tool results before full summarize when that alone frees enough budget.
- Intra-lite tool output bounds for non-MCP tools; MCP oversized results truncate-to-disk.
- OpenAI path **`tiktoken-rs`**: known models (`gpt-4o` → o200k, `gpt-4` / `gpt-3.5-turbo` → cl100k, etc.) use real BPE; unknown openai-compatible names and Anthropic keep the script-aware heuristic.
- Script-aware token heuristic (Latin vs CJK) as the non-tiktoken default.
- `/context` (alias `/tokens`) context report; preflight compact when estimate exceeds the hard window.
- Configurable Keel sandbox profiles: `--sandbox` / `ZENE_SANDBOX` / `[sandbox]` in config, plus `~/.zene/sandbox.toml` custom profiles (`off` | `workspace` | `read-only` | `strict` | custom).
- Host-side egress gating for `FetchUrl`, `WebSearch`, and HTTP MCP via Keel `check_egress`; `allow_hosts` allowlist support.
- Default credential path denies (read + Keel policy) for `~/.ssh`, `~/.aws`, `**/.env*`, `**/*.pem`, etc.; Read/Write prefer Keel `SpaceFs` when enforced.
- `[sandbox] auto_allow_bash` to skip Bash prompts while a sandbox profile is active.
- Docs: Cloudflare Pages pause guide; `deploy-web.sh` gated behind `ZENE_PAGES_DEPLOY=1`.

### Changed
- Default interactive entry is Web Agent (`zene` / `zene web`); `zene --tui` errors with a migration hint; debug line UI remains as `zene --repl`.
- ACP: bridge `tool_call` / `tool_call_update` / `plan` / `usage_update` / `current_mode_update` / `available_commands_update` / `agent_thought_chunk`; replay history on `session/load`; implement `session/list`, `session/close`, `session/set_mode`, `session/resume`; optional client FS + terminal bridges; FIFO prompt queue with in-turn cancel; correlate permission `toolCallId`; accept embedded prompt context; tighten JSON-RPC error codes.
- Stronger compaction ladder (reject thin summaries, tool-pair snap, sticky suppress after failed summarize).
- Compaction / water-level behavior tuned closer to grok-build Inter/Intra lite semantics.
- Landing page (`www/`) refreshed for current Zene features.
- GitHub Releases and `www/install.sh` publish/install both `zene` and `zene-gateway`; gateway serves UI with `Cache-Control: no-store`.

### Fixed
- Flaky `PreToolUse` hook test: ignore stdin BrokenPipe when the hook exits before reading payload.

## v0.1.6 (2026-07-18)

### Added
- Usage-driven context water level, full-replace compaction, and input ladder (`verbatim → fitted → lossy`).
- Permission modes: `default` / `accept_edits` / `dont_ask` / `bypass`, plus allow/deny/ask rules.
- Session recovery: compaction checkpoints, `/rewind`, `/fork`, `/session-info`, `/compact`.
- Background `Bash`/`Task` with `TaskOutput`; `zene --worktree` session git worktrees.
- MCP HTTP transport alongside stdio; `zene mcp doctor`.
- Headless `zene -p` with `--output-format json`.
- Minimal ACP stdio agent: `zene acp` (`initialize`, `session/*`, permission bridge).

### Changed
- LLM retry classification for overflow / rate-limit / transient errors.
- Grok-alignment roadmap items P1–P6 marked complete in `docs/ROADMAP.md` / `docs/ENGINE.md`.

## v0.1.5 (2026-05-31)

### Changed
- Default CLI startup to TUI; `--repl` for line REPL.
- TUI turn UX, model/provider configuration, and permission prompting improvements.

## v0.1.4 (2026-05-30)

### Added
- **WebSearch** and **FetchUrl** tools for DuckDuckGo search and page fetch.
- **Todo** tool with session-persisted todo lists (`TodoWrite` / store).
- **AskUser** collaboration tool; parallel tool execution via `tool_scheduler`.
- **Agent profiles** in config (`agent_profile`) for model/tool presets.
- Compaction **v2**: improved context trimming and token accounting.
- `docs/ENGINE.md` architecture notes.

### Fixed
- **unigateway-sdk** pinned to crates.io `2.1.1` (CI/release builds no longer need a local path).

### Changed
- Session records and ROADMAP/README updates for Batch 7 capabilities.
