# baodou Desktop MVP

这是 baodou 的 Rust + Tauri 2 + React/WebView 桌面端。Rust 是宿主和唯一的 UI IPC 边界，现有 Python 包仍作为本地 agent / capture / vision / llama 兼容运行时使用。

## 启动

```bat
cd /d D:\Projects\baodou\desktop
npm install
npm run tauri:dev
```

启动前需要保证：

- Rust stable、Node.js 22+、npm 可用；
- `conda env dev` 已存在；
- 项目根目录的 Python 依赖已经安装：`conda run -n dev python -m pip install -e "D:\Projects\baodou[dev]"`；
- live 推理仍使用项目根目录配置中的 `D:\llama`、oneAPI 和模型文件。

## MVP 行为

- 默认是 `mock + dry-run`，可以完整观察“输入目标 → 屏幕观察 → 计划等待确认 → 执行 → 结果”工作流，不会注入真实鼠标/键盘事件。
- 切换到真实桌面模式只改变观察链路的意图；真实 actuator 仍受 Python 配置、权限、风险策略和确认门控保护。
- `Rust host connected` 表示 WebView 与 Tauri Rust 宿主 IPC 已连接，不代表真实键鼠模式已经开启。
- 任务输入支持 Enter 发送，Shift+Enter 换行；右侧运行信息可查看模型、设备、UIA/Rules 和协议状态。
- 确认、暂停、停止均通过 Rust command 触发，React 不直接访问 Python、llama-server 或 actuator。

## 构建

```bat
npm run build
npm run tauri:build
```

Windows 安装包生成位置在 `src-tauri/target/release/bundle/`。第一次打包会下载 Tauri 和 WebView2 相关依赖。

## IPC 边界

MVP 使用 `protocol_version = 1.0.0`，当前命令包括：

- `get_runtime`
- `run_task`
- `confirm_task`
- `pause_runtime`
- `stop_runtime`

任务进度通过 `task-event` 事件发送。后续扩展字段必须保持 camelCase，并继续把任务计划、动作、确认、审计和错误作为结构化事件传输。
