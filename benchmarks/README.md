# 脱敏基准集与验收评测

本目录承载待办中的“验收指标与基准”，保证每项优化都用同一组基准截图与运行指标验收，
避免仅凭主观体验判断。

## 目录结构

```text
benchmarks/
  cases/
    manifest.json          # 基准集清单：场景、每张截图期望的事实 / 禁止内容
    artifacts/             # 脱敏截图（.png）。截图缺失时评测脚本会跳过并提示
  results/                 # run-<时间戳>.json / .csv 评测结果
  phase_a/                 # 历史阶段产物（服务端日志等）
scripts/bench_run.mjs      # 模型侧指标评测脚本（Node ≥18，无外部依赖）
```

## 合成脱敏截图

仓库内置 9 张完全合成的 1280×720 UI 场景，不包含真实账号、邮件、文件名或桌面数据。
在 Windows 上可确定性重建：

```powershell
.\scripts\generate_benchmark_fixtures.ps1
```

如需扩充真实场景，再按以下规则收集并脱敏：

1. 覆盖 9 类场景（待办要求至少 20–30 张，可在 `manifest.json` 中不断增加 case）：
   聊天/邮件、浏览器、IDE/终端、表格、弹窗与错误、多窗口、中文小字、暗色主题、滚动或视频。
2. 截图放入 `benchmarks/cases/artifacts/`，命名如 `chat-email-inbox.png`，与 `manifest.json`
   中的 `screenshotPath` 对应。
3. 每一类补充 `expectedKeyFacts`（期望的关键事实）、`allowedUncertainty`（允许的不确定表述）、
   `forbidden`（禁止的虚构 / 操作建议）。
4. 发布前对截图做脱敏（遮挡姓名、账号、会话内容等敏感信息）。

## 运行评测

先启动本地 llama-server（与 Baodou 相同配置，`-c` 不变），再执行：

```bash
# 默认使用 http://127.0.0.1:8765/v1/chat/completions，可用环境变量覆盖
BAODOU_LLAMA_URL=http://127.0.0.1:8765/v1/chat/completions node scripts/bench_run.mjs

# 只跑某个 case、与历史结果对比
node scripts/bench_run.mjs --case small-chinese
node scripts/bench_run.mjs --base benchmarks/results/run-xxx.json
```

输出为 `benchmarks/results/run-<时间戳>.json/.csv`，含：

- `firstTokenMs` / `firstContentTokenMs`：首个非空 `content` token 时延（≈ prefill + 发送；二者当前同值，后者语义更明确）
- `finishReason`：服务端结束原因（例如 `stop` / `length`），用于发现输出上限耗尽
- `firstReadableMs`：首个可读整句时延
- `finalMs` / `totalMs`：最终结果时延、请求总耗时
- `promptTokens` / `completionTokens`：usage token
- `factsHit / factsTotal`：关键事实命中率
- `forbiddenHits`：命中禁止词（操作建议 / 坐标 / ACTION）
- `jitter`：同图重复识别产生的语义改写次数（稳定性）
- `lowInfo`：模型是否明确表达了不确定性（“看不清”策略被触发）

应用侧指标（截图编码、prefill、首 token、生成、UI 渲染，以及 `skippedRounds/rounds`
跳帧率）由运行中的 Baodou 写入 `RuntimeSnapshot.metrics`，主窗口轮询可见；把同一次
运行结果归档到 `results/` 即形成前后对比表。

## 验收规则

- 同一张截图重复识别：主结论稳定，无同义反复改写或残句闪烁；
- 悬浮窗每轮有限次可读更新，最终内容完整；
- 正常场景不截断；不清晰场景明确表达不确定性而非虚构；
- 改动前后任一项关键指标（准确率 / 稳定性 / 时延）明显回归 → 不合入该优化。
