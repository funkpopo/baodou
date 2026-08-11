# Rust migration roadmap

## Completed foundation

- Tauri 2 desktop host and React workspace
- Rust screen capture through `xcap`
- Rust PNG/base64 image pipeline
- Rust llama-server client through `reqwest`
- Rust task state machine and structured single-step plan parser
- Rust risk, action and coordinate policy
- User confirmation, pause and stop through typed Tauri IPC

## Next implementation milestones

1. Add a Rust Windows UI Automation adapter and a stable `elementId` model.
2. Replace coordinate-only plans with UIA-first target relocation.
3. Add a Rust native input executor guarded by foreground-window, target freshness and per-step confirmation checks.
4. Add Rust llama-server supervision for `D:\llama`, including oneAPI environment setup, health checks and restart diagnostics.
5. Add tray lifecycle, single-instance ownership and global emergency-stop hotkey in Tauri/Rust.
6. Add replayable Rust fixtures and a small isolated Windows test surface after interactive MVP behavior is stable.

## Intentionally excluded

- Python modules, Conda environments and Python test infrastructure
- A Python bridge or Python sidecar for the desktop runtime
- Automatic execution of high-risk or irreversible actions
