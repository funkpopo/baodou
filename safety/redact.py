"""PII / secret redaction for logs, audit, and model context (Phase G).

Default: mask password-like fields, bank card numbers, mainland ID numbers,
and common secret patterns. Screen pixels are handled by capture.privacy.
"""

from __future__ import annotations

import re
from typing import Any

from core.logging import get_logger, log_event

_log = get_logger("safety.redact")

# 18-digit mainland ID (loose check; last digit may be X)
_RE_ID18 = re.compile(r"\b([1-9]\d{5})(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
# 15-digit legacy ID
_RE_ID15 = re.compile(r"\b([1-9]\d{7})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}\b")
# Bank / card: 13–19 digits with optional spaces/dashes (Luhn not required for mask)
_RE_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
# Password / secret style assignments in free text
_RE_PASSWORD_KV = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|密码|口令)\s*[=:：]\s*\S+"
)
# Email local-part keep domain for debugging? Mask full for safety.
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phone (CN mobile)
_RE_PHONE = re.compile(r"\b1[3-9]\d{9}\b")

_MASK = "[REDACTED]"


def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _mask_card_match(m: re.Match[str]) -> str:
    raw = re.sub(r"[^\d]", "", m.group(0))
    # Avoid masking plain years / short numbers already filtered by length
    if len(raw) < 13:
        return m.group(0)
    # Prefer Luhn-valid; still mask long digit runs that look like PANs
    if _luhn_ok(raw) or len(raw) >= 16:
        return _MASK
    return m.group(0)


def redact_text(text: str | None, *, enabled: bool = True) -> str:
    """Return redacted copy of free text. Empty/None → empty string."""
    if not text:
        return ""
    if not enabled:
        return text
    out = text
    out = _RE_PASSWORD_KV.sub(rf"\1={_MASK}", out)
    out = _RE_ID18.sub(_MASK, out)
    out = _RE_ID15.sub(_MASK, out)
    out = _RE_CARD.sub(_mask_card_match, out)
    out = _RE_PHONE.sub(_MASK, out)
    out = _RE_EMAIL.sub(_MASK, out)
    return out


def looks_like_secret(text: str | None) -> bool:
    if not text:
        return False
    if _RE_PASSWORD_KV.search(text):
        return True
    if _RE_ID18.search(text) or _RE_ID15.search(text):
        return True
    for m in _RE_CARD.finditer(text):
        raw = re.sub(r"[^\d]", "", m.group(0))
        if _luhn_ok(raw) or len(raw) >= 16:
            return True
    return False


def redact_mapping(
    data: dict[str, Any],
    *,
    enabled: bool = True,
    keys_always_mask: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Deep-ish redact of string values; always mask known secret keys."""
    if not enabled:
        return dict(data)
    secret_keys = keys_always_mask or frozenset(
        {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "image_b64",
            "raw_pixels",
            "ocr_raw",
        }
    )
    out: dict[str, Any] = {}
    for k, v in data.items():
        lk = str(k).lower()
        if lk in secret_keys or any(s in lk for s in ("password", "secret", "token", "b64")):
            out[k] = _MASK
        elif isinstance(v, str):
            out[k] = redact_text(v, enabled=True)
        elif isinstance(v, dict):
            out[k] = redact_mapping(v, enabled=True, keys_always_mask=secret_keys)
        elif isinstance(v, list):
            out[k] = [
                redact_mapping(i, enabled=True, keys_always_mask=secret_keys)
                if isinstance(i, dict)
                else (redact_text(i, enabled=True) if isinstance(i, str) else i)
                for i in v
            ]
        else:
            out[k] = v
    return out


def redact_for_log(event: str, fields: dict[str, Any], *, enabled: bool = True) -> dict[str, Any]:
    cleaned = redact_mapping(fields, enabled=enabled)
    if enabled and fields != cleaned:
        log_event(_log, "safety.redact", event=event, keys=list(fields.keys())[:12])
    return cleaned
