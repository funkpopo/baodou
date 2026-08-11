# 结构化协议（阶段 B）

**协议版本：** `PROTOCOL_VERSION = 1.0.0`（见 `core/models.py`）

模块之间**禁止**直接传递未约束的自然语言作为动作指令。自然语言仅出现在：

- 用户目标 `TaskContext.user_goal`
- 模型观察文本 `ScreenObservation.observation`（只读描述）
- 日志 `notes` / 审计摘要

所有可执行意图必须进入 `ActionPlan` / `ActionStep`。

## 核心类型

| 类型 | 用途 |
|---|---|
| `ScreenFrame` | 截图帧 + 尺寸/DPI/frame_id |
| `UIElement` / `UIVisionResult` | 识别元素列表 |
| `ScreenObservation` | 模型屏幕理解 |
| `InferenceRequest` / `InferenceResponse` | 推理往返 |
| `ActionPlan` / `ActionStep` | 任务计划与单步动作 |
| `ActionResult` | 执行结果 |
| `VerificationResult` | 动作后验证 |
| `SafetyDecision` | 安全裁决 |
| `TaskContext` | 任务状态机上下文（含 `corrections`） |
| `UserCorrection` / `CorrectionKind` | 用户目标修正（阶段 H） |
| `ActivityPhase` / `ActivityStatus` | UI 活动/隐私指示 |
| `MetricsSnapshot` | 延迟与资源快照 |
| `PipelineEvent` | 带 `trace_id` 的链路事件 |

## 坐标约定

- `BBox` / 点击目标默认使用**屏幕物理像素**（虚拟桌面，与鼠标一致）
- `ScreenFrame.dpi_scale` + `scale_x/y`：图像像素 ↔ 物理像素（阶段 C）
- `UIElement.bbox_logical` + `dpi_scale`：逻辑/DIP 备份，避免高 DPI 误点（阶段 D）
- 图像源检测（OCR/rules）必须经 `frame.image_to_screen` 再写入 `UIElement`

## 元素 ID

- 短期稳定 id：`{type_prefix}_{content_hash[:8]}`（如 `btn_a3f2c101`）；mock demo 保留 `btn_search_01`
- `content_hash`：type + 规范化文本 + 量化 bbox + role
- 绑定 `frame_id`；跨帧用 `element_stale` / `matches_hash` 判断失效
- 模型上下文优先 `element_id`，禁止优先依赖裸坐标

## 模型输出（阶段 E）

模型原始文本必须先经 `inference.parse` + `inference.validate`：

1. 提取完整 JSON（拒绝截断流）
2. 归一为 `observe_plan` / `observation` schema
3. 动作白名单、`element_id` 存在性、坐标物理 bounds
4. 通过后才写入 `InferenceResponse.observation` / `.plan`

非法输出：`ErrorCode.output_schema_invalid` 或降级为空 `ActionPlan`（`steps=[]`），**禁止**直达 actuator。

Prompt 版本见 `inference.prompts.PROMPT_VERSION`；变更需同步 `benchmarks/phase_e/fixtures/`。

## 风险与确认

```text
RiskLevel: low | medium | high
ActionStep.requires_confirmation: bool
SafetyDecision.allowed / requires_confirmation / blocked_by
```

高风险关键词（配置 `safety.sensitive_keywords`）会抬升为 `high` 并默认硬拦截。

## 任务状态机（阶段 F）

```text
TaskState:
  idle → observing → planning → awaiting_confirmation
       → executing → verifying → completed | failed | paused | cancelled
```

- `ActionPreview`：用户可见「做什么 / 目标在哪 / 预计影响」
- `StepRecord`：单步审计（preview → safety → action → verification → recovery）
- 定位优先 `element_id`；裸坐标需 `allow_coordinate_fallback` + 二次确认
- 执行前 `reidentify`；失败策略见 `RecoveryAction`（reidentify / skip / pause / fail）
- 默认 `actuator.dry_run=true`，合法计划也不得绕过确认直接注入系统输入

## 错误码

见 `core/errors.py` → `ErrorCode`，包括：

- `model_load_failed` / `inference_timeout` / `output_parse_failed`
- `capture_failed` / `vision_timeout`
- `permission_denied` / `risk_blocked` / `confirmation_required`
- `target_invalid` / `target_stale` / `action_failed` / `verification_failed`
- `cancelled`

异常基类：`BaodouError`，可 `to_dict()` 写入日志。

## 示例：UI 元素

```json
{
  "element_id": "btn_search_01",
  "type": "button",
  "role": "button",
  "text": "搜索",
  "bbox": { "x": 1080, "y": 40, "width": 96, "height": 36 },
  "bbox_logical": { "x": 720, "y": 27, "width": 64, "height": 24 },
  "confidence": 0.97,
  "clickable": true,
  "dpi_scale": 1.5,
  "content_hash": "a3f2c1b4d5e6",
  "source": ["uia", "ocr"],
  "frame_id": "frame-...",
  "needs_review": false
}
```

## 示例：动作计划

```json
{
  "goal": "点击搜索按钮",
  "steps": [
    {
      "action": "click",
      "target_element_id": "btn_search_01",
      "risk": "low",
      "requires_confirmation": true,
      "preconditions": ["element.visible == true"],
      "expected_change": "interact with btn_search_01"
    }
  ],
  "stop_if": ["target_missing", "window_changed"]
}
```
