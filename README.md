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
  └─ action loop     观察、决策、窗口激活、输入与结果验证
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

默认使用 `native computer use`：Rust 会反复采集主屏幕和动态窗口清单，由模型根据用户意图选择真实目标窗口；每轮生成并执行一个动作，再从新截图验证结果，直到目标完成或达到 12 步上限。

如果目标应用尚未显示，模型可以生成通用 `open_app` 动作，通过 Windows Search 启动应用。任务运行期间 Windows 前台会被 Computer Use 接管；可在任意应用中按 `Ctrl+Alt+Esc` 紧急中止。若用户主动切换前台窗口，待执行动作会被取消并重新规划。

## 本地模型模式

启动带视觉能力的 llama-server 后，在同一终端设置端点：

```powershell
$env:BAODOU_LLAMA_URL = "http://<host>:8765/v1/chat/completions"
npm run tauri:dev
```

应用会使用 Rust `reqwest` 直接发送屏幕 PNG 给 llama-server，每轮要求模型返回一个结构化动作或确认目标已经完成。动作可以先通过窗口标题激活其它 Windows 应用，再执行鼠标点击、文本输入或按键操作。baodou 本身会继续运行，任务状态和日志不会因目标窗口切换而丢失。

## 构建安装包

```bat
npm run build
npm run tauri:build
```

Windows 产物位于：

- `src-tauri/target/release/bundle/msi/`
- `src-tauri/target/release/bundle/nsis/`

更多信息见 [docs/architecture.md](docs/architecture.md) 和 [docs/development.md](docs/development.md)。
