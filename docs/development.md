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
| `src/` | React/TypeScript main launcher and floating spirit UI |
| `src-tauri/src/` | Rust recognition runtime and Tauri commands |
| `src-tauri/icons/` | Windows package icons |
| `model/` | Model metadata and local model placement notes |
| `data/` | Portable runtime data (`config.json`, `baodou.db`) created next to the exe |
| `docs/` | Product and technical documentation |

## Manual MVP check

1. Launch `npm run tauri:dev`.
2. Confirm the main window only shows the original spirit and a **启动** button.
3. Click **启动** and verify a transparent always-on-top floating spirit window appears.
4. Confirm recognition text refreshes in the floating bubble.
5. Click **停止** and verify recognition ends and the floating window hides.
6. Confirm `data/config.json` and `data/baodou.db` are created beside the running executable (dev build target directory).

No Python environment, Conda environment, or `pip install` command is required by this project.
