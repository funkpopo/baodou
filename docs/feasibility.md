# 阶段 A 技术可行性报告

> 状态：**阶段 A 完成**  
> 环境：Windows 11 · conda `dev` · `D:\llama` SYCL · Intel Arc A770 · oneAPI setvars  
> 日期：2026-08-11

---

## 1. 结论摘要

| 项 | 结论 |
|---|---|
| 目标 OS / 发布形态 | **Windows 11 x64 本地桌面应用**；不做 macOS/Linux 第一版 |
| 模型文件 | `model/Qwen3.5-2B-UD-Q4_K_XL.gguf`（Unsloth Dynamic Q4_K，约 1.28 GB） |
| 视觉能力 | **具备**。需配套 `model/mmproj-F16.gguf`（约 637 MB，CLIP / qwen3vl_merger） |
| 推理后端 | **`D:\llama\llama-server.exe`（ggml-sycl）+ Intel oneAPI**，设备 `SYCL0` |
| 禁止 | conda 中的 `llama-cpp-python` 官方 wheel（仅 `ggml-cpu`，无 SYCL/CUDA）作为生产推理 |
| 开发语言 | Python 3.12（conda `dev`）为主；推理为**独立进程**（HTTP OpenAI 兼容 API） |
| 阶段验收 | 本机可加载模型，对一张屏幕截图做稳定视觉推理，并得到可解析 JSON |

**总判断：阶段 A 技术路线可行。**  
视觉链路用 GGUF + mmproj 即可，不必另接独立视觉模型；但 2B 模型对复杂桌面的元素定位与坐标精度有限，后续必须由 UI Automation / OCR 提供结构化候选，模型侧重语义与规划。

---

## 2. 目标平台与发布形态

- **OS**：Windows 11（实测 `10.0.26200`）
- **形态**：本地常驻助手 + **独立 `llama-server` 推理进程**
- **多屏**：双 1920×1080 横排，虚拟桌面 3840×1080（已采集）
- **明确不做**：macOS / Linux 第一版打包与适配

---

## 3. 硬件基线

来源：`benchmarks/phase_a/results/hardware_baseline.json`

| 资源 | 实测 |
|---|---|
| CPU | 12th Gen Intel Core i5-12400（6C/12T） |
| 内存 | 31.7 GB |
| GPU | **Intel Arc A770 Graphics**（驱动 32.0.101.8864） |
| SYCL 设备 | `SYCL0: Intel(R) Arc(TM) A770 Graphics`（约 15932 MiB VRAM） |
| oneAPI | `D:\Intel\oneAPI\setvars.bat`（必须先 call，再启动 llama） |
| llama 构建 | `D:\llama`，含 `ggml-sycl.dll`、`mtmd.dll`、`llama-server.exe`（build b10356） |
| 屏幕 | 主屏 1920×1080 + 副屏 1920×1080 |

**GPU 强制策略：**

- 启动参数：`-dev SYCL0 -ngl 99 --mmproj-offload`
- 不允许默认落到 CPU；`llama-cpp-python` 的 `n_gpu_layers` 在本机 CPU wheel 上无效

---

## 4. 模型元数据与多模态格式

来源：`benchmarks/phase_a/results/model_metadata.json` 与模型 README

### 4.1 主模型

| 字段 | 值 |
|---|---|
| 架构 | `qwen35` |
| 名称 | Qwen3.5-2B |
| 量化 | UD-Q4_K_XL（Unsloth） |
| block_count | 24 |
| embedding | 2048 |
| 原生上下文 | **262144** tokens |
| MVP 运行 `n_ctx` | **4096**（避免 KV 过大；长任务再升） |
| Chat template | 内置 Jinja（含 image/video 分支，约 7.8k 字符） |
| pipeline | `image-text-to-text` |

### 4.2 视觉投影 mmproj

| 字段 | 值 |
|---|---|
| 文件 | `mmproj-F16.gguf` |
| `clip.has_vision_encoder` | true |
| projector | `qwen3vl_merger` |
| vision image_size | 768 |
| patch_size | 16 |
| projection_dim | 2048 |

### 4.3 图像输入方式（已验证）

`llama-server` OpenAI 兼容接口：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "Analyze this screenshot..."}
      ]
    }
  ],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

服务端 `modalities`: `{ "vision": true, "video": true, "audio": false }`。

**结论：当前 GGUF 可直接看图，无需额外视觉服务；必须同时加载 mmproj。**

---

## 5. 开发语言与进程模型

| 模块 | 建议 | 说明 |
|---|---|---|
| capture / ui_vision / agent / actuator / safety / frontend | Python 3.12 | conda env `dev` |
| inference | **独立进程** `D:\llama\llama-server` | SYCL GPU；崩溃隔离；可热重启 |
| 进程间协议 | HTTP ` /v1/chat/completions` | 后续可再加 gRPC/命名管道 |
| 备选 | 自编译带 SYCL 的 `llama-cpp-python` | 非必须；阶段 B 可评估 |

推荐边界：应用进程不内嵌 ggml；只做客户端。这样 oneAPI 环境变量与 GPU 驱动问题集中在 server 启动脚本。

---

## 6. 基线性能指标

来源：`benchmarks/phase_a/results/baseline_summary.json`（GPU 完整跑批，2026-08-11）

### 6.1 GPU 正式基线

| 指标 | 值 | 备注 |
|---|---|---|
| 后端 | `D:\llama` llama-server SYCL | `-dev SYCL0 -ngl 99` + mmproj |
| 截图（mss 主屏） | **62.7 ms** | 缩放至 1280×720 PNG |
| 验收视觉推理 | **12.8 s** | JSON 可解析，`vision_used=true` |
| 场景通过率 | **27/30 = 90%** | 文本 8/8，视觉 19/22 |
| 场景平均延迟 | **9.5 s** | min 1.5 s / max 13.0 s |
| 生成吞吐（典型） | ~47–49 tok/s | 2B Q4 on Arc A770 |
| prompt 吞吐（含图） | ~430–450 tok/s | |
| Server 进程 RSS | ~2.6–2.9 GB | 另占 Arc 显存（全层 offload） |
| 空闲 CPU（采样） | ~8.5% | 推理前 |

### 6.2 场景集（30 条）

- T01–T08：文本能力探针 — **全部通过**
- V01–V22：实时主屏截图视觉任务 — **19/22 通过**
- 失败项：
  - **V04**：输出被 markdown fence 截断，`no_json_object`
  - **V09 / V22**：JSON 中段非法（疑似截断或未转义引号）
- 脚本：`benchmarks/phase_a/run_baseline.py`
- 产物：`baseline_report.json` / `baseline_summary.json` / `artifacts/acceptance_raw.txt`

### 6.3 与 CPU 旧基线对比

| | CPU `llama-cpp-python` | GPU `D:\llama` SYCL |
|---|---|---|
| 验收 JSON | 失败（截断） | **成功** |
| 验收延迟 | ~52 s | **~13 s** |
| 场景通过 | 14/30 | **27/30** |
| 平均延迟 | ~34 s | **~9.5 s** |

**结论：生产路径必须使用 GPU SYCL server，禁止 CPU wheel 作为默认推理。**

---

## 7. 代表场景能力判断

| 能力 | 可行性 | 说明 |
|---|---|---|
| 读取页面/桌面概要 | 高 | 视觉摘要可用，中文 UI 可识别大意 |
| 识别按钮/输入框类型 | 中 | 能列候选，bbox 为近似值，不能当点击真值 |
| 判断输入框位置 | 中 | 需 UI Automation/OCR 校准 |
| 描述当前状态 | 高 | 适合 observation 字段 |
| 输出精确坐标 / 稳定 element_id | **低（单独靠模型）** | 必须由 `ui_vision` 提供 id+bbox，模型只选 id |
| 结构化 JSON 输出 | 中高 | 需 system schema + 解析/修复；易被 markdown fence 或截断影响 |
| 高风险操作判断 | 中 | 文本风险分级可用；执行层必须硬编码拦截 |

**失败与限制（已知）：**

1. `max_tokens` 过小会导致 JSON 截断 → 解析失败（需 ≥512–768，并做截断修复）  
2. 模型可能输出 ` ```json ` 围栏 → 客户端需剥离  
3. 思考模式偶发（应 `enable_thinking: false`）  
4. 2B 对密集 IDE/多窗口桌面易幻觉细节  
5. 坐标为模型估计，**禁止**直接驱动鼠标  
6. oneAPI `setvars` 在无 VS 环境下有告警，但不影响 SYCL 运行  
7. `AdapterRAM` WMI 读数不可靠（显示 ~2GB）；以 SYCL 报告的 16GB 为准  

---

## 8. 依赖与版本

| 组件 | 版本 / 路径 |
|---|---|
| Python | 3.12.13（conda-forge，env `dev`） |
| llama.cpp | b10356（0666ad2b2），`D:\llama`，Clang 20.1.8 Windows x86_64 |
| 后端 | ggml-sycl + Level Zero |
| 模型 | Qwen3.5-2B UD-Q4_K_XL + mmproj-F16 |
| 采集 | mss + Pillow |
| 系统信息 | psutil |
| GGUF 检查 | gguf |
| **不用** | `llama-cpp-python` 0.3.x CPU wheel 做 GPU 推理 |

启动模板：

```bat
call "D:\Intel\oneAPI\setvars.bat"
set PATH=D:\llama;%PATH%
llama-server.exe ^
  -m "D:\Projects\baodou\model\Qwen3.5-2B-UD-Q4_K_XL.gguf" ^
  --mmproj "D:\Projects\baodou\model\mmproj-F16.gguf" ^
  --mmproj-offload -ngl 99 -dev SYCL0 -c 4096 ^
  --host 127.0.0.1 --port 8765 -np 1 --jinja
```

基线一键：

```bat
conda activate dev
python benchmarks\phase_a\collect_hardware.py
python benchmarks\phase_a\inspect_model.py
python benchmarks\phase_a\run_baseline.py
```

---

## 9. 最终技术选型（阶段 A 锁定）

1. **OS**：仅 Windows 11 桌面  
2. **模型**：Qwen3.5-2B GGUF Q4 + **mmproj-F16** 多模态  
3. **推理**：`D:\llama` SYCL **`llama-server`**，Arc **GPU**，独立进程  
4. **应用语言**：Python 3.12 模块化；推理客户端 HTTP  
5. **视觉主路径**：截图 →（阶段 D）UI 识别 → 紧凑元素 JSON + 可选缩略图 → Qwen  
6. **操作主路径**：模型只输出 `element_id` / 计划；执行前校验与确认  
7. **安全**：默认只读；点击/输入需确认；高风险硬拦截  

---

## 10. 阶段 A 验收对照

| 验收项 | 状态 |
|---|---|
| 本机加载指定模型 | ✅ `llama-server` + SYCL0 + mmproj |
| 一张屏幕截图稳定推理 | ✅ 验收 12.8 s，JSON 合法，视觉启用 |
| 可解析结构化结果 | ✅ system schema + 客户端解析/修复 |
| 若无视觉则定补充链路 | ✅ 已具备 mmproj，**无需**另建视觉服务 |
| 硬件与延迟基线 | ✅ hardware / model / baseline JSON |
| 20–30 场景 | ✅ **27/30（90%）**，失败均为 JSON 格式边角，非“不能看图” |

---

## 11. 进入阶段 B 的前置建议

1. 把 `llama-server` 启停封装为正式 `inference` 服务模块与配置项  
2. 统一 `ScreenObservation` / `ActionPlan` Pydantic（或 JSON Schema）  
3. 日志带 trace_id：frame → vision → llm → plan  
4. 不要在 B 阶段实现点击；先 mock actuator  
5. 为 Arc/SYCL 写健康检查：`llama-cli --list-devices` 必须出现 `SYCL0`

---

## 12. 产物索引

| 路径 | 说明 |
|---|---|
| `benchmarks/phase_a/collect_hardware.py` | 硬件 + SYCL 设备 |
| `benchmarks/phase_a/inspect_model.py` | GGUF/mmproj 元数据 |
| `benchmarks/phase_a/run_baseline.py` | GPU 截图 + 场景基线 |
| `benchmarks/phase_a/results/*.json` | 数值结果 |
| `benchmarks/phase_a/artifacts/primary_screenshot.png` | 验收截图 |
| `benchmarks/phase_a/artifacts/acceptance_raw.txt` | 验收原始输出 |
| `benchmarks/phase_a/artifacts/start_server.bat` | oneAPI + server 启动 |
| `docs/feasibility.md` | 本文件 |
