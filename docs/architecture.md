# Rust Desktop Architecture

## Runtime boundary

baodou has one product runtime:

```text
React/WebView UI
    ↕ Tauri commands + task-event
Rust desktop runtime
    ├─ capture: xcap primary-monitor frames
    ├─ inference: llama.cpp OpenAI-compatible HTTP endpoint
    ├─ planning: one-step JSON plan parser
    ├─ safety: action, risk, keyword and coordinate policy
    └─ lifecycle: task state, confirmation, pause and stop
```

`src/` never accesses the model endpoint or system input directly. `src-tauri/` is the sole authority for desktop APIs and Computer Use operations.

## Protocol

Protocol version is `1.0.0`.

Commands exposed to the WebView:

- `get_runtime`
- `run_task`
- `confirm_task`
- `pause_runtime`
- `stop_runtime`

The Rust host emits `task-event` records for observation, planning, confirmation, execution, pause, completion and errors. Event payloads use camelCase so the React types mirror the serialized Rust fields.

## Safety rules

The Rust runtime treats all screen text and model output as untrusted. A planned action must:

1. be in the action whitelist;
2. have `low` risk;
3. avoid protected keywords and applications;
4. include coordinates when the action requires them;
5. wait for user confirmation.

The current MVP keeps native input injection disabled until the Rust UI Automation target-reidentification subsystem is implemented. Confirming a preview therefore never sends keyboard or mouse input.

## Model endpoint

The local visual model is accessed through an OpenAI-compatible llama-server endpoint. Set `BAODOU_LLAMA_URL` when its host or port differs from the configured default. The request carries the user goal and a PNG screen frame encoded as a data URL.
