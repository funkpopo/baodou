# baodou

本地 AI 电脑操作辅助助手（Windows 11）。

边缘模型：`Qwen3.5-2B-GGUF` + `mmproj`，推理后端：`llama.cpp` SYCL（Intel Arc）。  
UI 识别与模型解耦：识别提供 `element_id` + bbox，模型负责语义与规划，执行层强制校验与确认。

## 当前进度

| 阶段 | 状态 |
|---|---|
| A 可行性 / 基线 | ✅ 完成（见 `docs/feasibility.md`） |
| **B 项目骨架** | ✅ 本目录结构、协议、配置、日志、mock 链路 |
| C+ 采集与后续 | ⏳ 未开始 |

## 快速开始（阶段 B，纯 mock）

```bat
cd /d D:\Projects\baodou
conda activate dev
python -m pip install -e ".[dev]"
pytest
python -m frontend.cli demo --goal "点击搜索按钮"
```

更多命令见 [`docs/setup.md`](docs/setup.md)。

## 模块

| 目录 | 职责 |
|---|---|
| `capture/` | 屏幕帧 |
| `ui_vision/` | `UIElement[]` |
| `inference/` | llama-server 客户端 / mock |
| `agent/` | 任务与计划 |
| `actuator/` | 结构化动作执行 |
| `safety/` | 风险、确认、拦截 |
| `frontend/` | CLI |
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

阶段 B **不会**向系统注入真实鼠标/键盘事件（`actuator.dry_run: true`）。  
高风险关键词默认硬拦截。
