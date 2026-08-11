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
2. enumerate the current foreground window and visible window titles;
3. ask the model to infer the target window from user intent and live inventory;
4. execute one supported action when more work is needed;
5. wait briefly and capture a new frame for verification.

Click coordinates returned against the resized model image are mapped back to the physical display before input injection. A stop request is checked before every observation and again immediately before every action.

The runtime has no built-in application alias table. If the main planner omits an action, a separate intent-resolution request may select only a title present in the current window inventory. Invented or stale titles are rejected.

Model generation is not constrained to JSON. The model protocol uses short `STATUS`, `OBSERVATION`, `ACTION`, `TARGET`, `TEXT`, and `EXPECTED` lines because small local vision models follow it more reliably.

If no visible window matches but the goal clearly names an application, the intent resolver may return an `open_app` query. The executor opens Windows Search, enters that model-derived query, launches the result, and returns to the observation loop. Before non-window actions, the runtime compares the foreground window with the one captured during planning; a user focus change cancels that action and triggers a fresh observation. `Ctrl+Alt+Esc` is monitored natively as an emergency stop even when baodou is not focused.

## Model endpoint

The local visual model is accessed through an OpenAI-compatible llama-server endpoint. Set `BAODOU_LLAMA_URL` when its host or port differs from the configured default. The request carries the user goal and a PNG screen frame encoded as a data URL.
