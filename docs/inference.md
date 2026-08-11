# 阶段 E：llama.cpp 推理层与 Qwen 适配

> 状态：**阶段 E 完成**  
> 环境：Windows 11 · conda `dev` · `D:\llama` SYCL · oneAPI `D:\Intel\oneAPI\setvars.bat` · Intel Arc  
> 日期：2026-08-11

---

## 1. 固定运行时记录

| 项 | 值 |
|---|---|
| 二进制 | `D:\llama\llama-server.exe` |
| 版本 | **10356** (`0666ad2b2`) |
| 编译器 | Clang 20.1.8 · Windows x86_64 |
| 后端 | **ggml-sycl**（`ggml-sycl.dll` 等） |
| 视觉 | `mtmd` / mmproj（`--mmproj` + `--mmproj-offload`） |
| Chat template | 模型内置 Jinja（`--jinja`） |
| GPU 策略 | **强制** `-dev SYCL0 -ngl 99`；禁止默认 CPU |
| 模型 | `model/Qwen3.5-2B-UD-Q4_K_XL.gguf` + `model/mmproj-F16.gguf` |
| MVP `n_ctx` | 4096 |
| `n_batch` | 512 |
| `max_tokens` | 768 |
| `temperature` | 0.3 |
| `enable_thinking` | false |
| Prompt 版本 | `PROMPT_VERSION`（见 `inference/prompts.py`） |

探测命令：

```bat
python -m frontend.cli infer info --json-out benchmarks/phase_e/results/runtime.json
```

---

## 2. 模块

| 文件 | 职责 |
|---|---|
| `inference/server.py` | 进程生命周期：加载 / 预热 / 停止 / 恢复 |
| `inference/http_client.py` | OpenAI 兼容客户端：多模态请求、重试、流式门控 |
| `inference/prompts.py` | 版本化 system prompt、停止词、动作白名单 |
| `inference/schema.py` | JSON Schema / GBNF + 线格式模型 |
| `inference/parse.py` | JSON 提取 / 截断修复 / 流式完整判定 |
| `inference/validate.py` | schema、元素存在、坐标范围、动作白名单 |
| `inference/degrade.py` | 忙/超时/非法输出降级（空计划 + 最近可信观察） |
| `inference/runtime_info.py` | 版本与依赖记录 |
| `inference/mock.py` | 无 GPU 回归路径（同样走 validate） |

---

## 3. 生命周期

```text
ensure_ready / start
    call oneAPI setvars
    → llama-server -m … --mmproj … -ngl 99 -dev SYCL0 -c 4096 --jinja
    → /health
    → warmup (短 completion)
observe / stream_observe
    → cancel token 可中断等待
stop / recover
    → 仅停止本进程托管的 server；外部已启动实例不强制杀死
```

```bat
python -m frontend.cli infer server status
python -m frontend.cli infer server start
python -m frontend.cli infer server warmup
python -m frontend.cli infer server stop
```

配置：

- `inference.auto_start_server`：CLI/`--start-server` 可打开
- `inference.warmup_on_start`
- `inference.server_start_timeout_sec`（默认 180）

---

## 4. 统一请求接口

输入统一为：

1. **用户目标** `user_goal`
2. **结构化 UI 摘要**（`ui_vision.context.serialize_*`，优先 `element_id`）
3. **可选截图**（`image_url` data URL，PNG base64）

模式：

| mode | 说明 |
|---|---|
| `observe_plan` | 观察 + 可选低风险 `ActionPlan` 草图 |
| `observation` | 只读观察 |

服务端约束（可选）：`constraint_mode = json_schema | grammar | none`  
客户端**始终**校验；非法输出**不会**带着可执行 plan 进入操作层。

---

## 5. 流式门控

`stream_observe` 会累积 token，仅当：

1. JSON 括号完整（`is_json_complete`）
2. 解析成功
3. `validate_model_output` 通过  

才设置 `ready_for_action=True`。  
**注意：** 即便 `ready_for_action`，仍须经阶段 F 安全/确认后才能执行；本阶段 CLI **不注入键鼠**。

---

## 6. 校验与降级

拒绝条件（fatal → 无 plan）：

- 非法 / 不在白名单的 `action`
- `target_element_id` 不在当前 `UIVisionResult`
- 裸坐标越出屏幕物理 bounds
- 输出截断（`finish_reason=length` / 不完整 JSON）
- schema 无法归一

降级（`degrade_on_error: true`）：

- 返回 `ok=False` + 最近可信观察或 UI-only 摘要
- `plan.steps = []`，要求用户确认 / 人工介入
- **绝不**把 raw 自然语言当系统输入

---

## 7. Prompt 版本与回归

- 改 system / 模板必须 bump `PROMPT_VERSION`
- 样例：`benchmarks/phase_e/fixtures/`
- 单测：`tests/test_inference_parse_validate.py`、`tests/test_inference_mock.py`
- Bench：`python benchmarks/phase_e/run_inference_bench.py --mock`
- Live：`python benchmarks/phase_e/run_inference_bench.py --http --start-server`

---

## 8. CLI

```bat
python -m frontend.cli infer info
python -m frontend.cli infer prompts
python -m frontend.cli infer once --backend mock --vision-backend mock --goal "描述当前屏幕"
python -m frontend.cli infer once --backend http --start-server --goal "描述当前屏幕"
python -m frontend.cli infer once --backend http --stream --goal "点击搜索"
```

环境变量：`BAODOU_INFERENCE=http`、`BAODOU_LLAMA_HOST/PORT`、`BAODOU_N_CTX`、`BAODOU_DEVICE` 等。

---

## 9. 验收对照（阶段 E）

| 要求 | 状态 |
|---|---|
| 固定 llama 版本/编译/依赖 | ✅ `infer info` + 本文 |
| 生命周期 load/warmup/infer/cancel/release/recover | ✅ `server.py` |
| n_ctx / threads / batch / ngl / KV 相关配置与基准钩子 | ✅ config + phase_e bench |
| 文本 + UI 摘要 + 图像统一接口 | ✅ `HttpInference.observe` |
| Qwen chat template / system / stop / max tokens | ✅ `--jinja` + prompts |
| JSON schema/grammar + 协议校验 | ✅ schema + validate |
| 流式但完整校验后才可进入操作层 | ✅ `stream_observe` 门控 |
| 元素/坐标/白名单校验 | ✅ validate |
| 超时/取消/重试/降级 | ✅ http_client + degrade |
| prompt 版本与回归样例 | ✅ PROMPT_VERSION + fixtures |

**阶段验收：** 输入截图/UI 状态与用户任务后，模型可输出合法观察或计划草图；非法、越权、不完整输出被拒绝或降级，**不会**直接触发系统输入。
