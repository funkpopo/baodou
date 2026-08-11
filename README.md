# baodou

本地 AI 电脑操作助手，使用 Rust、Tauri 2、系统 WebView 和 React 构建。

桌面应用的在线运行路径完全由 Rust 承担：屏幕采集、图像编码、llama-server HTTP、任务状态、计划解析、风险门控与 Tauri IPC 均不依赖 Python。

## 技术结构

```text
src/                 React + TypeScript 工作台 UI
      ↕ typed Tauri command / task-event
src-tauri/           Rust 桌面宿主与 Computer Use runtime
  ├─ xcap            屏幕采集
  ├─ image/base64    图像转换
  ├─ reqwest         llama-server 客户端
  ├─ serde           结构化计划协议
  └─ safety policy   风险、动作、坐标和确认门控
model/               本地模型说明与元数据（模型权重不提交）
docs/                架构、开发与运行文档
```

## 启动开发环境

前置条件：Windows 11、Rust stable、Node.js 22+、npm，以及 WebView2 Runtime。

```bat
cd /d D:\Projects\baodou
npm install
npm run tauri:dev
```

默认使用 `native preview + dry-run`：Rust 会采集主屏幕并生成低风险只读计划，但不会注入鼠标或键盘输入。

## 本地模型模式

启动带视觉能力的 llama-server 后，在同一终端设置端点：

```powershell
$env:BAODOU_LLAMA_URL = "http://<host>:8765/v1/chat/completions"
npm run tauri:dev
```

应用会使用 Rust `reqwest` 直接发送屏幕 PNG 给 llama-server，要求模型返回一个结构化、可确认的单步计划。高风险、非白名单和不完整坐标的动作会在 Rust 安全策略中暂停。

## 构建安装包

```bat
npm run build
npm run tauri:build
```

Windows 产物位于：

- `src-tauri/target/release/bundle/msi/`
- `src-tauri/target/release/bundle/nsis/`

更多信息见 [docs/architecture.md](docs/architecture.md) 和 [docs/development.md](docs/development.md)。
