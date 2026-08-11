# 阶段 D：UI 识别框架

## 目标

- 统一 `UIElement` 协议（类型、文本、物理 bbox、DPI、置信度、可交互性、来源）
- 多来源识别：`uia`（优先）+ `rules`（轻量视觉）+ 可选 `ocr`
- 去重融合、层级、ROI、稳定短期 `element_id` / `content_hash` / 失效判断
- 为 Qwen 提供紧凑上下文：筛选、编号框、`element_id` 优先（禁止裸坐标优先）
- **多分辨率正确性**：任意 DPI/缩放/图像降采样下，元素中心可映射回截图像素且与物理鼠标坐标一致

## 模块

| 文件 | 职责 |
|---|---|
| `ui_vision/base.py` | `UIRecognizer` 插件接口 + `UIVisionBackend` |
| `ui_vision/ids.py` | 稳定 id、content_hash、staleness |
| `ui_vision/coords.py` | 图像/物理/逻辑坐标换算 |
| `ui_vision/uia.py` | Windows UI Automation（comtypes） |
| `ui_vision/ocr.py` | OCR（可选 pytesseract） |
| `ui_vision/rules.py` | 边缘矩形启发式（按钮/输入框/图标） |
| `ui_vision/fuse.py` | 多源 IoU 融合、置信度、层级 |
| `ui_vision/context.py` | 紧凑 JSON/文本、目标筛选、编号框标注 |
| `ui_vision/pipeline.py` | 多源编排 `CompositeUIVision` |
| `ui_vision/factory.py` | `mock` / `composite` / 单源工厂 |
| `ui_vision/mock.py` | 合成树（按物理分辨率缩放） |

## 坐标约定（多分辨率）

```text
UIA BoundingRectangle  ──►  物理像素 bbox（进程 Per-Monitor DPI Aware）
OCR / rules 图像框      ──►  frame.image_to_screen  ──►  物理像素 bbox
物理 bbox              ──►  / dpi_scale            ──►  bbox_logical（DIP）
执行点击               ──►  只用物理 center / element_id
```

- `UIElement.bbox`：**虚拟桌面物理像素**（与 `GetCursorPos` 一致）
- `UIElement.bbox_logical` + `dpi_scale`：保留逻辑坐标，避免 125%/150%/200% 缩放误点
- 截图经 `max_width/height` 缩放后，`ScreenFrame.scale_x/y` 负责图像↔物理互转

## 配置（`config/default.yaml` → `ui_vision`）

```yaml
ui_vision:
  backend: composite          # mock | composite | uia | ocr | rules
  sources: [uia, rules]       # 可加 ocr
  confidence_threshold: 0.5
  max_elements: 64
  timeout_ms: 2000
  fuse_iou_threshold: 0.45
  context_max_elements: 32
  annotate_boxes: true
```

环境变量：`BAODOU_UI_VISION=mock|composite|uia|…`

可选 OCR：

```bat
pip install -e ".[ocr]"
rem 并安装 Tesseract OCR，保证 PATH 可用
```

## 命令

```bat
conda activate dev
cd /d D:\Projects\baodou

python -m frontend.cli vision once
python -m frontend.cli vision once --backend mock --goal "点击搜索"
python -m frontend.cli vision once --backend composite --goal "确定" --annotate benchmarks\phase_d\artifacts\ann.png
python -m frontend.cli vision once --json-out benchmarks\phase_d\results\once.json
python -m frontend.cli vision context --backend mock --goal "搜索"

python benchmarks\phase_d\run_vision_bench.py
pytest tests\test_ui_vision_ids_fuse.py tests\test_ui_vision_coords_context.py -q
```

## 紧凑上下文示例

模型应使用 `element_id`，不要手写坐标：

```json
{
  "element_id": "btn_a3f2c101",
  "type": "button",
  "text": "搜索",
  "bbox": {"x": 1720, "y": 40, "width": 96, "height": 36},
  "center": {"x": 1768, "y": 58},
  "confidence": 0.95,
  "clickable": true,
  "needs_review": false,
  "source": ["uia", "ocr"]
}
```

标注图：`annotate_from_frame` 绘制 `[序号:element_id]` 框，与 compact 列表顺序一致。

## 验收对照

| 项 | 验证 |
|---|---|
| 元素协议完整 | `UIElement` 含 type/text/bbox/center/confidence/flags/source/hash |
| UIA 可用 | `vision once --backend composite` 返回 `sources_used` 含 uia |
| 融合去重 | 单测 `test_fuse_merges_uia_and_ocr` |
| 多分辨率坐标 | 参数化单测 + bench `multi_res_mock.*.coord_ok` |
| 真机坐标 | bench `live.coord_audit.coord_ok`；center → `screen_to_image` 落在帧内 |
| 模型上下文 | `serialize_for_model` / `vision context`；标注图落盘 |
| 失效判断 | `element_stale` / `CompositeUIVision.is_stale` |

## 明确不做（本阶段）

- 大模型推理与 grammar（阶段 E）
- 真实键鼠点击（仍 dry-run；阶段 F）
- 完整安全策略与威胁建模（阶段 G）
- 重型检测网络（可后续以插件替换 `rules`）
