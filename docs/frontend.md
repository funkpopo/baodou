# 阶段 H：用户界面与可观测性

> 状态：**阶段 H 完成**  
> 环境：Windows 11 · conda `dev` · 默认 **dry_run** + **mock**  
> 日期：2026-08-11

---

## 1. 目标

为采集 → 识别 → 推理 → 计划 → 确认 → 执行 提供**常驻主窗口**与**可测试的会话层**：

| 能力 | 说明 |
|---|---|
| 主窗口控制 | 开始 / 暂停 / 继续 / 紧急停止 / 复位 |
| 任务输入 | 自然语言目标 |
| 屏幕解读 | 观察文本 + 元素列表 + 目标高亮 |
| 计划与确认 | 步骤预览、风险级别、确认/拒绝 |
| 用户修正 | 「不是这个」「点这个」「忽略区域」「点这里」→ 结构化任务上下文 |
| 可观测性 | 采集/识别/模型延迟、队列、CPU/内存、GPU 标签、最近错误 |
| 开发者诊断 | frame / UI 树 / 计划 / 预览 / 校验 / prompt 版本 |
| 隐私指示 | 采集中 / 识别中 / 推理中 / 即将执行 / 执行中 可见状态 |

**设计原则：** 业务逻辑在 `UISession`（无头），Tk 窗口只做渲染与输入；CLI `ui run` 与 GUI 共用同一会话。

---

## 2. 模块

| 文件 | 职责 |
|---|---|
| `frontend/session.py` | `UISession`：任务生命周期、确认桥、活动相位、快照 |
| `frontend/app.py` | Tk 主窗口 `BaodouApp` |
| `frontend/corrections.py` | `UserCorrection` 应用：改写 goal、过滤元素 |
| `frontend/metrics.py` | `MetricsCollector`：延迟与资源 |
| `frontend/highlight.py` | 元素框 / 目标十字准星 / 预览缩放 |
| `frontend/diagnostics.py` | 开发者诊断包（脱敏） |
| `frontend/cli.py` | `ui open|run|status|correct` |

协议扩展（`core/models.py`）：

- `CorrectionKind` / `UserCorrection`
- `ActivityPhase` / `ActivityStatus`
- `MetricsSnapshot`
- `TaskContext.corrections`

---

## 3. 活动 / 隐私指示

| `ActivityPhase` | 用户可见含义 |
|---|---|
| `idle` | 空闲 |
| `capturing` | 正在采集屏幕 |
| `recognizing` | 正在识别 UI |
| `inferring` | 正在推理 / 规划 |
| `awaiting_confirm` | **即将执行**，等待确认 |
| `executing` / `verifying` | 正在执行 / 验证 |
| `paused` / `stopped` / `error` | 暂停 / 紧急停止 / 错误 |

主窗口顶栏同步显示 📷 采集 / 🔍 识别 / 🧠 推理 / ⚠️ 即将执行 / 🖱️ 执行中。

---

## 4. 用户修正

修正进入 `TaskContext.corrections`，并拼入规划用 goal 的 `[用户修正]` 段；**不会**变成裸系统输入。

| 操作 | `CorrectionKind` | 效果 |
|---|---|---|
| 不是这个 | `reject_element` | 从候选元素中剔除 |
| 点这个 | `prefer_element` | 排序优先 |
| 点这里 | `click_here` | 记录物理坐标提示 |
| 忽略区域 | `ignore_region` | 过滤落在区域内的元素 |
| 备注 | `note` | 附加自然语言约束 |

```bat
python -m frontend.cli ui correct --goal "点击搜索"
python -m frontend.cli ui run --goal "点击搜索" --reject btn_x --prefer btn_search_01 --preview-only
```

---

## 5. 命令

```bat
conda activate dev

REM 主窗口（默认 mock + dry_run）
python -m frontend.cli ui open
python -m frontend.app
baodou-ui

REM 真实采集/识别预览（仍 dry_run，不注入键鼠）
python -m frontend.cli ui open --live

REM 无头会话（CI / 验收）
python -m frontend.cli ui status
python -m frontend.cli ui run --goal "点击搜索按钮" --yes --preview-only
python -m frontend.cli ui run --goal "点击搜索按钮" --yes

REM Bench
python benchmarks\phase_h\run_ui_bench.py
```

---

## 6. 配置

`config/default.yaml` → `frontend.*`：

```yaml
frontend:
  mode: cli                 # cli | gui
  window_title: "baodou — AI 桌面助手"
  refresh_ms: 500
  preview_max_width: 720
  preview_max_height: 480
  show_diagnostics: true
  metrics_interval_ms: 1000
  activity_indicators: true
  recent_errors_max: 12
  auto_refresh_preview: false
  default_mock: true
```

---

## 7. 安全边界（与 G/F 一致）

- 默认 `actuator.dry_run: true`，GUI 默认 mock 后端
- 确认回调不能绕过 `SafetyPolicy` / 高风险硬拦截
- 紧急停止走 `safety.control` + 全局 cancel token
- 诊断视图省略 `image_b64` 等大字段；审计仍走 `logs/audit/`

---

## 8. 验收

| 项 | 验证 |
|---|---|
| 主窗口可启动 | `ui open`（人工）或 `ui status`（自动化） |
| 无头任务 | `ui run --yes --preview-only` → plan + metrics |
| 用户修正 | `ui correct` / unit tests |
| 活动指示 | stop → `stopped`；reset → `idle` |
| 诊断 | `diagnostics()` 含 elements / plan / metrics |
| Bench | `run_ui_bench.py` 全 case PASS |

单元测试：`tests/test_frontend_session.py`
