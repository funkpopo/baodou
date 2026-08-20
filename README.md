# baodou

本地 AI 实时屏幕识别助手，使用 Rust、Tauri 2、系统 WebView 和 React 构建。

桌面应用的在线运行路径完全由 Rust 承担：屏幕采集、JPEG 轻量编码、画面变化检测、llama-server HTTP、低延迟识别、悬浮窗推送与 Tauri IPC 均不依赖 Python。

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

识别循环针对秒级反馈优化：静态画面会跳过重复模型请求；截图会先进行无损 PNG 压缩，再降至 768×432 供视觉模型处理，并流式显示首段结果。每轮识别独立完成，不读取或整理历史记忆，以保持最低响应延迟。实际首帧延迟取决于本机 GPU、llama.cpp 构建和模型加载状态。

运行时不含鼠标、键盘、窗口激活、应用启动或其他电脑控制能力。

## 用户自定义推理服务

项目不内置、发现、选择或推荐任何模型，也不要求固定目录结构。请在应用的配置页填写：

- 推理服务可执行文件路径；
- 主模型文件路径；
- 多模态投影文件路径；
- OpenAI 兼容接口地址；
- 由用户决定的 offload、batch、Flash Attention 等运行参数。

这些值保存到可执行文件旁的 `data/config.json`，应用只使用用户保存的路径，不会根据项目目录自动替换或迁移模型。

也可在启动前通过环境变量提供这些值：

```powershell
$env:BAODOU_LLAMA_SERVER = "<path-to-server>"
$env:BAODOU_MODEL_PATH = "<path-to-model>"
$env:BAODOU_MMPROJ_PATH = "<path-to-mmproj>"
$env:BAODOU_LLAMA_URL = "http://<host>:8765/v1/chat/completions"
npm run tauri:dev
```

## 便携式发行

数据库与配置保存在可执行文件旁的 `data/` 目录，便于整包拷贝分发。推理服务与模型文件由用户自行管理，应用不复制或打包它们。

```bat
npm run build
npm run tauri:build
```

Windows 产物位于：

- `src-tauri/target/release/baodou.exe`
- `src-tauri/target/release/bundle/nsis/`

更多信息见 [docs/architecture.md](docs/architecture.md) 和 [docs/development.md](docs/development.md)。
