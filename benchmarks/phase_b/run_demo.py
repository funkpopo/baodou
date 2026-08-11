"""Phase B acceptance helper: run mock pipeline and write results JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.logging import setup_logging
from core.pipeline import MockPipeline

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    setup_logging(
        level="INFO", json_logs=True, log_dir=cfg.app.log_dir, project_root=cfg.project_root
    )
    pipe = MockPipeline(cfg)

    goals = [
        "描述当前屏幕",
        "点击搜索按钮",
        "向搜索框输入文字",
        "删除文件并支付",  # expect block
    ]
    reports = []
    for goal in goals:
        result = pipe.run(goal)
        reports.append(
            {
                "goal": goal,
                "ok": result.ok,
                "trace_id": result.trace_id,
                "task_state": result.task.state.value,
                "event_kinds": [e.kind.value for e in result.events],
                "error": result.error.to_dict() if result.error else None,
                "elapsed_ms": result.elapsed_ms,
            }
        )

    out = RESULTS / "phase_b_demo.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for row in reports:
        print(
            f"- {row['goal']!r}: state={row['task_state']} ok={row['ok']} trace={row['trace_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
