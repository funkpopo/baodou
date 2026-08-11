# 环境启动与本地命令（阶段 B）

## 前置条件

| 项 | 说明 |
|---|---|
| OS | Windows 11 x64 |
| Python | **conda env `dev`**（Python 3.12） |
| 推理二进制 | **`D:\llama`**（`llama-server.exe`，ggml-sycl） |
| oneAPI | **`D:\Intel\oneAPI\setvars.bat`**（启动 server 前必须 call） |
| 模型（可选，阶段 E） | `model/Qwen3.5-2B-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf` |

阶段 B **默认使用 mock 后端**，无需 GPU / llama-server 即可跑通骨架与测试。

## 安装

```bat
cd /d D:\Projects\baodou
conda activate dev
python -m pip install -e ".[dev]"
```

仅运行时依赖：

```bat
python -m pip install -e .
```

## 常用命令

| 目的 | 命令 |
|---|---|
| 格式化 | `ruff format .` |
| 静态检查 | `ruff check .` |
| 类型检查（可选） | `mypy core capture ui_vision inference agent actuator safety frontend` |
| 单元测试 | `pytest` |
| 覆盖率 | `pytest --cov --cov-report=term-missing` |
| 版本 | `python -m frontend.cli version` |
| 查看配置 | `python -m frontend.cli config-show` |
| 健康检查 | `python -m frontend.cli health` |
| 探测 llama-server | `python -m frontend.cli health --http` |
| **Mock 全链路 demo** | `python -m frontend.cli demo --goal "点击搜索按钮"` |
| 只读观察 demo | `python -m frontend.cli demo --goal "描述当前屏幕"` |
| 写出链路 JSON | `python -m frontend.cli demo --json-out benchmarks/phase_b/results/demo.json` |
| 显示器列表 | `python -m frontend.cli capture monitors` |
| 单次截图 | `python -m frontend.cli capture once --mode primary` |
| 实时流（有界队列） | `python -m frontend.cli capture stream --seconds 2` |
| 采集 bench | `python benchmarks\phase_c\run_capture_bench.py` |

安装 entry points 后也可用：

```bat
baodou demo --goal "点击搜索按钮"
baodou-demo
```

## 配置

- 默认：`config/default.yaml`
- 覆盖路径：`BAODOU_CONFIG=path\to\file.yaml` 或 `--config`
- 常用环境变量：
  - `BAODOU_LOG_LEVEL=DEBUG`
  - `BAODOU_INFERENCE=mock|http`
  - `BAODOU_CAPTURE=mock|mss`
  - `BAODOU_LLAMA_HOST` / `BAODOU_LLAMA_PORT`
  - `BAODOU_N_CTX` / `BAODOU_N_GPU_LAYERS` / `BAODOU_DEVICE`

关键配置项：模型路径、`n_ctx`、线程/GPU layers、截图 FPS、识别阈值、确认策略、日志级别。详见 `config/default.yaml`。

## 日志与 Trace

- 默认 JSON 结构化日志（`app.log_json: true`），并写入 `logs/baodou.log`
- 每次 pipeline 运行生成 `trace_id`（`tr-…`）
- 同一 `trace_id` 串联：`frame` → `vision` → `inference` → `plan` → `safety` → `action` → `verification`

示例过滤（PowerShell）：

```powershell
Get-Content logs\baodou.log | Select-String "tr-"
```

## 优雅退出

- `Ctrl+C`（SIGINT）触发全局 `CancellationToken`
- 流水线各阶段调用 `token.check()`，取消后状态为 `cancelled`
- 二次 Ctrl+C 强制退出

## 模块独立 mock 运行

```bat
pytest tests/test_pipeline_mock.py::test_modules_independent_with_mock -q
python -m frontend.cli demo --goal "描述当前屏幕"
```

各包入口：

| 包 | Mock 类 | 职责 |
|---|---|---|
| `capture` | `MockCapture` | 合成帧元数据 |
| `ui_vision` | `MockUIVision` | 固定 UIElement[] |
| `inference` | `MockInference` | 结构化 ScreenObservation |
| `agent` | `MockAgent` | 规则 ActionPlan |
| `safety` | `SafetyPolicy` | 风险/确认/硬拦截 |
| `actuator` | `MockActuator` | dry-run 执行 + 验证 |
| `frontend` | CLI | 启动与展示 |

**阶段 B 不执行真实键鼠。** `actuator.dry_run` 必须为 `true`。

## 故障排查

1. `ModuleNotFoundError` → 确认在仓库根目录且已 `pip install -e .`
2. `配置文件不存在` → 检查 `config/default.yaml`
3. `health --http` 失败 → 先按 `docs/feasibility.md` 启动 `llama-server`（阶段 E 才强制）
4. 测试失败 → `pytest -vv` 查看断言
