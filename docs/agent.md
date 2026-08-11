# 阶段 F：任务代理与操作状态机

> 状态：**阶段 F 完成**  
> 环境：Windows 11 · conda `dev` · 默认 **dry_run**（不注入真实键鼠）  
> 日期：2026-08-11

---

## 1. 目标

把阶段 C/D/E 的采集、识别、推理结果接到**可追踪的任务闭环**：

```text
自然语言目标
  → observing（截图 + UI 识别 + 模型观察）
  → planning（短步骤计划）
  → awaiting_confirmation（预览：做什么 / 点哪里 / 影响）
  → executing（执行前重新识别定位）
  → verifying（执行后验证）
  → completed | failed | paused | cancelled
```

---

## 2. 状态机

| 状态 | 含义 |
|---|---|
| `idle` | 未开始 |
| `observing` | 采集 + 识别 + 观察 |
| `planning` | 生成 `ActionPlan` |
| `awaiting_confirmation` | 等待用户确认预览 |
| `executing` | 执行当前步骤 |
| `verifying` | 动作后验证 |
| `completed` / `failed` / `paused` / `cancelled` | 终态或可恢复暂停 |

实现：`agent/state.py`（非法迁移抛 `BaodouError`）。

```bat
python -m frontend.cli agent states
```

---

## 3. 模块

| 文件 | 职责 |
|---|---|
| `agent/runtime.py` | `TaskAgent` 主编排 |
| `agent/state.py` | 状态迁移 |
| `agent/preview.py` | 用户可见动作预览 |
| `agent/recovery.py` | 失败恢复策略 |
| `agent/planner.py` | mock / inference 规划 |
| `agent/mock.py` | 规则多步计划 |
| `actuator/relocate.py` | 执行前按 `element_id` 重定位 |
| `actuator/verify.py` | 动作后验证 |
| `actuator/win_input.py` | Windows `SendInput`（仍受 `dry_run` 门控） |
| `actuator/rate_limit.py` | 动作频率限制 |
| `actuator/factory.py` | `mock` \| `win` |

---

## 4. 动作与优先级

基础动作（`ActionType`）：

`move` · `click` · `double_click` · `right_click` · `drag` · `scroll` · `type` · `key` · `hotkey` · `wait` · `reidentify`

**定位优先级：**

1. `target_element_id`（当前帧精确匹配）
2. `content_hash` / 类型+文本+IoU 模糊重定位
3. 裸坐标：仅当 `allow_coordinate_fallback` 且二次确认（`coordinate_requires_confirm`）

页面变化或目标消失 → **暂停**，不盲目点击旧坐标。

---

## 5. 确认与安全

- 默认 `actuator.dry_run: true`：只走校验/预览/日志，不注入系统输入
- 默认需要确认；CLI `--yes` 打开 `agent.auto_confirm`（仍拦截高风险）
- `safety.policy` 高风险关键词硬拦截（删除/支付/转账/密码等）
- 预览字段：`summary`、目标 bbox/中心点、`expected_impact`、`warnings`

---

## 6. 失败恢复

`agent/recovery.py` 策略顺序：

1. 目标失效 → `reidentify`（有限次数）→ 仍失败则 `pause`
2. 可选步骤（`optional`）→ `skip_step`
3. 执行/验证失败 → 重试定位或暂停请求用户
4. 否则 `fail`

---

## 7. CLI

```bat
conda activate dev

REM 安全 mock 全链路（默认）
python -m frontend.cli agent run --goal "点击搜索按钮" --yes --mock
python -m frontend.cli agent preview --goal "点击搜索按钮"
python -m frontend.cli agent run --goal "输入 hello" --yes --mock
python -m frontend.cli agent run --goal "描述当前屏幕" --yes --mock

REM 仅预览，不执行
python -m frontend.cli agent run --goal "点击确定" --preview-only --mock

REM 真实采集/识别，但仍 dry_run（不注入键鼠）
python -m frontend.cli agent run --goal "描述当前屏幕" --live --yes
```

写出完整结果：

```bat
python -m frontend.cli agent run --goal "点击搜索按钮" --yes --mock --json-out benchmarks/phase_f/results/once.json
```

---

## 8. 配置（`config/default.yaml`）

```yaml
agent:
  backend: mock            # mock | inference
  auto_confirm: false
  reidentify_before_action: true
  max_recovery_attempts: 2
  pause_on_target_missing: true
  pause_on_verify_fail: true
  prefer_element_id: true
  coordinate_requires_confirm: true

actuator:
  backend: mock            # mock | win
  dry_run: true            # 默认禁止真实输入
  max_actions_per_minute: 30
```

环境变量：`BAODOU_AGENT` · `BAODOU_ACTUATOR` · `BAODOU_DRY_RUN` · `BAODOU_AUTO_CONFIRM`

---

## 9. 验收

```bat
pytest tests/test_agent_runtime.py -q
python benchmarks/phase_f/run_agent_bench.py
```

验收标准：

- [x] 状态机覆盖 idle→…→completed/failed/paused
- [x] ≥3 个低风险端到端任务（点击 / 输入 / 只读观察），每步可追踪目标、执行结果、验证结果
- [x] 目标消失时暂停，不盲目点击
- [x] 执行前重新识别；预览可见
- [x] 默认 dry_run，高风险拦截

---

## 10. 明确不做（留给 G/H）

- 完整威胁建模与热键紧急停止（阶段 G）
- 图形化确认 UI / 目标高亮窗（阶段 H）
- 默认关闭 dry_run 的无人值守真实点击
