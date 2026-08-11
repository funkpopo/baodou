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
| `TaskContext` | 任务状态机上下文 |
| `PipelineEvent` | 带 `trace_id` 的链路事件 |

## 坐标约定

- `BBox` / 点击目标默认使用**屏幕物理像素**
- 逻辑坐标与 DPI 在 `ScreenFrame.dpi_scale` 中保留，阶段 C 完善换算

## 元素 ID

- 短期稳定 id，例如 `btn_search_01`
- 绑定 `frame_id`；跨帧需重新识别（阶段 D 做 hash/失效）

## 风险与确认

```text
RiskLevel: low | medium | high
ActionStep.requires_confirmation: bool
SafetyDecision.allowed / requires_confirmation / blocked_by
```

高风险关键词（配置 `safety.sensitive_keywords`）会抬升为 `high` 并默认硬拦截。

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
  "confidence": 0.97,
  "clickable": true,
  "source": ["mock"],
  "frame_id": "frame-..."
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
