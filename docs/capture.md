# 阶段 C：屏幕采集与实时帧管线

## 目标

- 全屏（主屏 / 虚拟全桌面）、指定窗口、指定区域采集
- 多显示器、DPI/缩放、虚拟桌面原点；图像坐标 ↔ 物理屏幕坐标可互转
- 缩放 / 色彩 / 压缩 / ROI，降低下游数据量
- 帧变化检测：无明显变化时可不推 vision/model
- 有界队列 + 丢帧，推理变慢时内存不涨
- 区分 preview / vision / model / verify 四类流
- 敏感区域遮罩（手动 + 隐私窗口标题）

## 模块

| 文件 | 职责 |
|---|---|
| `capture/geometry.py` | 显示器列表、DPI、区域解析、坐标换算 |
| `capture/window_win.py` | Windows 窗口枚举 / 查找 |
| `capture/mss_backend.py` | mss 真机截图 |
| `capture/preprocess.py` | 缩放、色彩、编码、ROI |
| `capture/change.py` | 降采样帧差 |
| `capture/queue.py` | 有界队列（prefer newest） |
| `capture/privacy.py` | 敏感区域遮罩 |
| `capture/pipeline.py` | 多流实时管线 |
| `capture/frame.py` | `FramePacket`（meta + PIL，可 release） |
| `capture/factory.py` | `mock` / `mss` 工厂 |

## 坐标约定

- **物理像素**：虚拟桌面坐标系（与 `GetCursorPos` / 鼠标一致，进程已设 Per-Monitor DPI Aware）
- `ScreenFrame.origin_x/y`：本帧图像 (0,0) 对应的物理原点
- `scale_x/y`：图像像素 → 物理像素的倍率（含缩放）
- API：`frame.image_to_screen(ix, iy)` / `frame.screen_to_image(sx, sy)`

## 配置（`config/default.yaml` → `capture`）

- `backend: mss`（阶段 C 默认；测试可用 `BAODOU_CAPTURE=mock`）
- `mode: primary | all | window | region`
- `monitor_index`、`region`、`window_title` / `window_hwnd`
- `queue_size`、`drop_policy: newest`
- `change_threshold`、`streams.preview|vision|model|verify`
- `privacy.manual_masks` / `privacy_window_titles`

## 命令

```bat
conda activate dev
cd /d D:\Projects\baodou

python -m frontend.cli capture monitors
python -m frontend.cli capture once --mode primary --kind vision
python -m frontend.cli capture once --mode all --out benchmarks/phase_c/artifacts/all.png
python -m frontend.cli capture once --mode region --region 100,100,800,600
python -m frontend.cli capture once --window-title "记事本"
python -m frontend.cli capture stream --seconds 2

python benchmarks\phase_c\run_capture_bench.py
pytest tests\test_capture_mss.py tests\test_geometry.py tests\test_change_queue_privacy.py -q
```

## 验收对照

| 项 | 验证 |
|---|---|
| 单屏/多屏截图 | `capture once --mode primary/all` |
| 坐标一致 | `image_to_screen` 与 cursor round-trip；bench 写 `center_screen` |
| 队列不失控 | `capture stream` → `queues_bounded: true`；`high_watermark <= queue_size` |
| 变化检测 | `ChangeDetector` 单测；vision `only_on_change` |
| 隐私遮罩 | `privacy.apply_masks` 单测 + 配置 manual_masks |

## 明确不做（本阶段）

- 真实键鼠点击（仍 dry-run）
- UI Automation / OCR（阶段 D）
- 模型推理（阶段 E）
