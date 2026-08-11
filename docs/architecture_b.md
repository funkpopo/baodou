# 阶段 B 架构说明

## 目录

```text
baodou/
├── capture/          # 屏幕采集
├── ui_vision/        # UI 识别
├── inference/        # 模型客户端（mock / HTTP）
├── agent/            # 任务规划
├── actuator/         # 动作执行（mock dry-run）
├── safety/           # 风险与确认策略
├── frontend/         # CLI 入口
├── core/             # 模型、配置、日志、错误、取消、pipeline
├── config/           # default.yaml
├── tests/            # 单元 / 链路测试
├── benchmarks/       # 阶段评测（A 已有，B demo 结果可落 phase_b）
├── docs/             # 文档
├── model/            # GGUF（本地大文件）
└── pyproject.toml    # 构建、ruff、pytest、mypy
```

## 数据流（mock 全链路）

```text
user_goal
   → Capture.capture()           → ScreenFrame          [event: frame]
   → UIVision.recognize()        → UIVisionResult       [event: vision]
   → Inference.observe()         → ScreenObservation    [event: inference]
   → Agent.plan()                → ActionPlan           [event: plan]
   → Safety.evaluate()           → SafetyDecision       [event: safety]
   → Actuator.execute()          → ActionResult         [event: action]
   → Actuator.verify()           → VerificationResult   [event: verification]
   → DONE / ERROR / CANCEL
```

全程共享 `trace_id`（线程上下文 + 模型字段）。

## 进程模型（与阶段 A 选型一致）

- **应用进程**：Python 模块（采集/识别/代理/安全/前端）
- **推理进程**（阶段 E 启用）：`D:\llama\llama-server` SYCL，HTTP OpenAI 兼容 API
- 阶段 B 默认 `inference.backend: mock`，`HttpInference` 仅作骨架

## 取消与退出

- `core.cancel.CancellationToken` 全局单例
- CLI 安装 SIGINT/SIGTERM 处理器
- pipeline 各 hop 调用 `check()`

## 明确不做（阶段 B）

- 真实截图管线（阶段 C）
- 真实 UI Automation / OCR（阶段 D）
- 完整 prompt / grammar（阶段 E）
- 真实键鼠注入（阶段 F）
- 完整安全威胁建模（阶段 G）
