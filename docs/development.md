# Development

## Commands

```bat
npm install
npm run tauri:dev
npm run build
npm run tauri:build
```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/` | React/TypeScript desktop interface |
| `src-tauri/src/` | Rust task runtime and Tauri commands |
| `src-tauri/icons/` | Windows package icons |
| `model/` | Model metadata and local model placement notes |
| `docs/` | Product and technical documentation |

## Manual MVP check

1. Launch `npm run tauri:dev`.
2. Enter a concrete task such as “切换到记事本并输入测试文本”.
3. Confirm that the event timeline cycles through observing, planning, executing, and reaches “任务已验证完成”.
4. Verify that the target window is activated, input is delivered, and completion occurs only after a fresh screenshot confirms the result.
5. Change foreground focus during planning and verify that the pending action is skipped and the desktop is observed again.
6. Press `Ctrl+Alt+Esc` while another app is active and confirm that no later action is injected.

No Python environment, Conda environment, or `pip install` command is required by this project.
