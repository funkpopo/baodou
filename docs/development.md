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
2. Enter a normal low-risk task such as “描述当前屏幕”.
3. Confirm that the event timeline reaches “计划已准备，等待确认”.
4. Confirm the step and verify that preview mode reports no system input.
5. Optionally launch llama-server, set `BAODOU_LLAMA_URL`, and enable local model mode to validate the visual request path.

No Python environment, Conda environment, or `pip install` command is required by this project.
