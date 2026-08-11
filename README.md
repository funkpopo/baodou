# baodou

本地 AI 电脑操作辅助助手（Windows 11）。

边缘模型：`Qwen3.5-2B-GGUF` + `mmproj`，推理后端：`llama.cpp` SYCL（Intel Arc）。  
UI 识别与模型解耦：识别提供 `element_id` + bbox，模型负责语义与规划，执行层强制校验与确认。

## 当前进度

| 阶段 | 状态 |
|---|---|
| A 可行性 / 基线 | ✅ 完成（见 `docs/feasibility.md`） |
| B 项目骨架 | ✅ 协议、配置、日志、mock 链路 |
| C 屏幕采集 | ✅ mss 多模式采集、坐标、有界队列、四类帧流 |
| D UI 识别 | ✅ UIA + rules 融合、多分辨率坐标、紧凑上下文 |
| E 推理层 | ✅ llama-server 生命周期、结构化校验、降级与 prompt 版本 |
| F 任务代理 | ✅ 状态机、预览确认、重定位、验证、失败暂停（默认 dry_run） |
| G 安全权限 | ✅ 风险分级、硬拦截、紧急停止、审计脱敏、威胁门控 |
| **H 用户界面** | ✅ 主窗口、修正、指标、诊断、活动/隐私指示 |
| I+ 评测与发布 | ⏳ 未开始 |

## 快速开始

```bat
cd /d D:\Projects\baodou
conda activate dev
python -m pip install -e ".[dev]"
pytest
python -m frontend.cli demo --goal "点击搜索按钮"
python -m frontend.cli capture once --mode primary
python -m frontend.cli vision once --goal "点击搜索"
python -m frontend.cli capture stream --seconds 2
python -m frontend.cli infer once --backend mock --vision-backend mock --goal "描述当前屏幕"
python -m frontend.cli agent run --goal "点击搜索按钮" --yes --mock
python -m frontend.cli safety status
python -m frontend.cli safety check --goal "删除并支付" --action click
python -m frontend.cli ui status
python -m frontend.cli ui run --goal "点击搜索按钮" --yes --preview-only
python -m frontend.cli ui open
```

更多：[`docs/setup.md`](docs/setup.md) · [`docs/capture.md`](docs/capture.md) · [`docs/ui_vision.md`](docs/ui_vision.md) · [`docs/inference.md`](docs/inference.md) · [`docs/agent.md`](docs/agent.md) · [`docs/safety.md`](docs/safety.md) · [`docs/frontend.md`](docs/frontend.md)

## 模块

| 目录 | 职责 |
|---|---|
| `capture/` | 屏幕帧 |
| `ui_vision/` | `UIElement[]` |
| `inference/` | llama-server 客户端 / mock |
| `agent/` | 任务与计划 |
| `actuator/` | 结构化动作执行 |
| `safety/` | 风险、确认、拦截、审计、脱敏、紧急停止 |
| `frontend/` | CLI + GUI 会话 / 主窗口 |
| `core/` | 协议、配置、日志、错误、取消 |
| `tests/` | 自动化测试 |
| `benchmarks/` | 性能与场景基线 |
| `docs/` | 文档 |

## 协议与配置

- 数据模型：`core/models.py`（`PROTOCOL_VERSION`）
- 错误码：`core/errors.py`
- 默认配置：`config/default.yaml`
- 协议说明：`docs/protocol.md`

## 安全提示

默认 **不会**向系统注入真实鼠标/键盘事件（`actuator.dry_run: true`）。  
默认安全模式 **只读**（`safety.default_mode: read_only`）；高风险关键词硬拦截。  
`--yes` 仅自动确认低风险，不能绕过权限/白名单/紧急停止。详见 `docs/safety.md`。
