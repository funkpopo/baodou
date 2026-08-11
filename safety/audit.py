"""Local non-repudiable audit trail (Phase G).

Default: JSONL under logs/audit/. No raw pixels, no secrets.
Provides disable-persistence and cleanup helpers.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import AppConfig, SafetySection
from core.logging import get_logger, log_event
from core.models import AuditEventKind, AuditRecord

from safety.redact import redact_mapping

_log = get_logger("safety.audit")


def _audit_dir(cfg: SafetySection, project_root: Path) -> Path:
    base = Path(cfg.audit_dir)
    if not base.is_absolute():
        base = project_root / base
    return base


class AuditLog:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg = config.safety
        self.root = config.project_root
        self._path: Path | None = None
        if self.cfg.audit_enabled:
            d = _audit_dir(self.cfg, self.root)
            d.mkdir(parents=True, exist_ok=True)
            day = datetime.now(UTC).strftime("%Y%m%d")
            self._path = d / f"audit-{day}.jsonl"

    @property
    def path(self) -> Path | None:
        return self._path

    def enabled(self) -> bool:
        return bool(self.cfg.audit_enabled and self._path is not None)

    def record(
        self,
        kind: AuditEventKind | str,
        *,
        trace_id: str = "",
        task_id: str = "",
        summary: str = "",
        payload: dict[str, Any] | None = None,
        model_version: str = "",
        prompt_version: str = "",
    ) -> AuditRecord:
        kind_e = kind if isinstance(kind, AuditEventKind) else AuditEventKind(kind)
        raw_payload = payload or {}
        # Never persist frames / OCR / full model context unless explicitly enabled
        cleaned = dict(raw_payload)
        if not self.cfg.persist_frames:
            for k in list(cleaned.keys()):
                if "frame" in k.lower() and k not in ("frame_id", "frame_ids"):
                    if k.endswith("_id") or k.endswith("_ids"):
                        continue
                    cleaned.pop(k, None)
            cleaned.pop("image_b64", None)
            cleaned.pop("image_path", None)
        if not self.cfg.persist_ocr:
            cleaned.pop("ocr", None)
            cleaned.pop("ocr_text", None)
            cleaned.pop("ocr_raw", None)
        if not self.cfg.persist_model_context:
            cleaned.pop("prompt", None)
            cleaned.pop("messages", None)
            cleaned.pop("raw_text", None)
            cleaned.pop("model_context", None)

        if self.cfg.redact_pii:
            cleaned = redact_mapping(cleaned, enabled=True)
            summary = redact_mapping({"s": summary}, enabled=True).get("s", summary)

        rec = AuditRecord(
            kind=kind_e,
            trace_id=trace_id,
            task_id=task_id,
            summary=summary[:500],
            payload=cleaned,
            model_version=model_version,
            prompt_version=prompt_version,
        )
        log_event(_log, "safety.audit", **rec.log_summary())
        if self.enabled() and self._path is not None:
            try:
                line = json.dumps(rec.model_dump(mode="json"), ensure_ascii=False, default=str)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as exc:
                log_event(_log, "safety.audit_write_failed", error=str(exc), level=40)
        return rec

    def cleanup(self, *, older_than_days: int | None = None, wipe_all: bool = False) -> dict[str, Any]:
        """Delete audit files. wipe_all removes the audit directory contents."""
        d = _audit_dir(self.cfg, self.root)
        removed: list[str] = []
        if not d.exists():
            return {"removed": [], "dir": str(d)}
        if wipe_all:
            for p in d.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)
                    removed.append(p.name)
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(p.name + "/")
            self.record(
                AuditEventKind.CLEANUP,
                summary=f"wipe_all audit dir ({len(removed)} entries)",
                payload={"removed": removed[:50]},
            )
            return {"removed": removed, "dir": str(d), "wipe_all": True}

        cutoff = None
        if older_than_days is not None and older_than_days >= 0:
            cutoff = datetime.now(UTC).timestamp() - older_than_days * 86400
        for p in d.glob("audit-*.jsonl"):
            try:
                if cutoff is not None and p.stat().st_mtime >= cutoff:
                    continue
                if older_than_days is None:
                    continue
                p.unlink(missing_ok=True)
                removed.append(p.name)
            except OSError:
                continue
        self.record(
            AuditEventKind.CLEANUP,
            summary=f"cleanup older_than_days={older_than_days}",
            payload={"removed": removed},
        )
        return {"removed": removed, "dir": str(d), "older_than_days": older_than_days}

    def disable_persistence(self) -> None:
        """Runtime switch: stop writing audit files (in-memory/log only)."""
        self.cfg.audit_enabled = False
        self._path = None
        log_event(_log, "safety.audit_disabled")
