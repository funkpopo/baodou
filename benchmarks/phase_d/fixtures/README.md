# Phase D UI vision fixtures / eval notes

Offline-friendly cases exercised by unit tests + bench:

| Case | Covered by | Notes |
|---|---|---|
| Normal controls (button/input/window) | `MockUIVision`, fuse tests | Stable demo ids |
| Multi-resolution 1080p / 125% / 150% / 2K | `test_image_screen_bbox_roundtrip_multi_res`, bench `multi_res_mock` | Physical + logical boxes |
| Scaled capture (`scale_x/y` ≠ 1) | coord tests | OCR/rules image→screen |
| Goal filtering | `filter_elements_for_goal` | Prefer 搜索 over 取消 |
| UIA+OCR merge | `test_fuse_merges_uia_and_ocr` | Text from OCR, type from UIA |
| Hierarchy | `test_hierarchy_parent_child` | window → button |
| Stale target | `test_element_stale_detection` | moved / missing |
| Self-drawn rect (rules) | `test_rules_recognizer_finds_button_like_rect` | No UIA |
| Live desktop | `run_vision_bench.py` | Real UIA tree + coord audit |

Future expansions (not blocking Phase D): dark theme screenshots, tiny fonts, occlusion, remote desktop captures under `artifacts/`.
