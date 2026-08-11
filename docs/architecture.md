# Rust Desktop Architecture

## Runtime boundary

baodou has one product runtime:

```text
React/WebView UI
    ↕ Tauri commands + task-event
Rust desktop runtime
    ├─ capture: xcap primary-monitor frames
    ├─ inference: llama.cpp OpenAI-compatible HTTP endpoint
    ├─ intent: dynamic foreground and visible-window resolution
    ├─ planning: repeated observe-act-verify tagged-line protocol
    ├─ actions: dynamic app launch, window activation, click, text and keyboard input
    └─ lifecycle: task state, verification, pause and stop
```

`src/` never accesses the model endpoint or system input directly. `src-tauri/` is the sole authority for desktop APIs and Computer Use operations.

## Protocol

Protocol version is `1.0.0`.

Commands exposed to the WebView:

- `get_runtime`
- `run_task`
- `pause_runtime`
- `stop_runtime`

The Rust host emits `task-event` records for observation, planning, execution, verification, completion and errors. Event payloads use camelCase so the React types mirror the serialized Rust fields.

## Computer Use loop

Each task runs for at most 12 rounds:

1. capture the latest primary-monitor frame;
2. enumerate the current foreground window and visible *actionable application* titles;
3. ask the model to infer the target window from user intent and live inventory;
4. execute one supported action when more work is needed;
5. wait briefly and capture a new frame for verification.

Click coordinates returned against the resized model image are mapped back to the physical display before input injection. A stop request is checked before every observation and again immediately before every action.

The runtime has no built-in application alias table. If the main planner omits an action, a separate intent-resolution request may select only a title present in the current window inventory. Invented or stale titles are rejected.

The inventory is observational rather than a hand-maintained allow/deny list: all titled visible top-level windows may be reported, including shell and utility surfaces. On the first round, a separate target-acquisition pass binds the user goal to either a real inventory title or a dynamic Windows Search query. It chooses a window only with positive task relevance; the main visual planner does not get to promote an arbitrary visible surface into a target. Every activation is then checked against the live inventory and verified by the resulting foreground HWND. `SetForegroundWindow` is only a request and its return value is not treated as proof of success. The native executor restores the window, temporarily attaches input threads when needed, brings it to the top, and polls the foreground HWND for a bounded interval before allowing the next action.

An explicit launch verb in the user's task (`打开`, `启动`, or `运行`) is a stronger intent constraint than the current window inventory. In that case the harness extracts the application query from the task and starts that application directly; it does not ask the target-acquisition model to choose an arbitrary visible window first. This prevents a task such as “打开浏览器” from being redirected to a desktop or utility surface while still avoiding a system-window blacklist.

Model generation is not constrained to JSON. The model protocol uses short `STATUS`, `OBSERVATION`, `ACTION`, `TARGET`, `TEXT`, and `EXPECTED` lines because small local vision models follow it more reliably.

If no visible window matches but the goal clearly names an application, the intent resolver may return an `open_app` query. The executor opens Windows Search, enters that model-derived query, launches the result, and returns to the observation loop. Before non-window actions, the runtime compares the foreground window with the one captured during planning; a user focus change cancels that action and triggers a fresh observation. `Ctrl+Alt+Esc` is monitored natively as an emergency stop even when baodou is not focused.

For local-model reliability, each request is intentionally limited to one action and a short tagged-line protocol. The model is given the live inventory and current foreground window, while the executor owns coordinate scaling, input injection, target validation, and completion verification. This keeps platform-sensitive behavior deterministic and avoids spending model tokens on a multi-step plan that may become stale after the first action.

Text entry is modeled as a state transition rather than an unconditional append. The protocol supports `clear_search_bar` and `replace`; the latter performs select-all, deletion, and text entry as one executor-owned operation. This makes retries idempotent and prevents stale browser form contents from being concatenated with the new task query. Plain `input` remains available only for intentional append behavior.

## Model endpoint

The local visual model is accessed through an OpenAI-compatible llama-server endpoint. Set `BAODOU_LLAMA_URL` when its host or port differs from the configured default. The request carries the user goal and a PNG screen frame encoded as a data URL.
