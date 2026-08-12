# baodou

本地 AI 实时屏幕识别助手，使用 Rust、Tauri 2、系统 WebView 和 React 构建。

桌面应用的在线运行路径完全由 Rust 承担：屏幕采集、图像编码、llama-server HTTP、低延迟识别、悬浮窗推送与 Tauri IPC 均不依赖 Python。

## 技术结构

```text
src/                 React + TypeScript 启动台 / 悬浮精灵 UI
      ↕ typed Tauri command / recognition-event
src-tauri/           Rust 桌面宿主与实时识别 runtime
  ├─ xcap            屏幕采集
  ├─ image/base64    图像转换
  ├─ reqwest         llama-server 客户端
  ├─ rusqlite        便携式本地数据库
  └─ recognition loop 截屏、视觉识别、结果去重与悬浮窗刷新
data/                运行后生成：config.json + baodou.db（相对 exe）
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

主窗口只提供 **启动** 按钮。启动后：

1. Rust 以默认关注点开始实时屏幕识别；
2. 显示系统级独立、无边框、置顶且不出现在任务栏的悬浮精灵窗；
3. 识别结果在精灵旁消息气泡中实时刷新；
4. 点击 **停止** 后识别结束，悬浮窗自动隐藏。

运行时不含鼠标、键盘、窗口激活、应用启动或其他电脑控制能力。

## 本地模型模式

将 `llama-server.exe` 与模型文件放在可执行文件同级目录（或按 `data/config.json` 中的路径配置），应用会自动尝试拉起本地视觉服务：

```text
<portable root>/
  llama-server.exe
  model/
    Qwen3.5-2B-UD-Q4_K_XL.gguf
    mmproj-F16.gguf
  data/
    config.json
    baodou.db
```

也可在启动前手动指定接口：

```powershell
$env:BAODOU_LLAMA_URL = "http://<host>:8765/v1/chat/completions"
npm run tauri:dev
```

## 便携式发行

数据库与配置保存在可执行文件旁的 `data/` 目录，便于整包拷贝分发。后续可将模型、配置与宿主整合进同一发行目录，形成 portable 应用。

```bat
npm run build
npm run tauri:build
```

Windows 产物位于：

- `src-tauri/target/release/baodou.exe`
- `src-tauri/target/release/bundle/nsis/`

更多信息见 [docs/architecture.md](docs/architecture.md) 和 [docs/development.md](docs/development.md)。
