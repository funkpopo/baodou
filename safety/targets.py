"""Allow / deny lists for apps, window titles, and domains (Phase G)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from core.config import SafetySection
from core.models import ActionStep, ScreenFrame, UIElement, UIVisionResult


@dataclass(frozen=True)
class TargetCheck:
    allowed: bool
    reason: str
    blocked_by: str | None = None
    matched: str = ""


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _match_any(value: str, patterns: list[str]) -> str | None:
    """Substring match (case-insensitive). Returns matched pattern or None."""
    v = _norm(value)
    if not v:
        return None
    for p in patterns:
        if p and _norm(p) in v:
            return p
    return None


def _extract_domains_from_text(text: str) -> list[str]:
    found: list[str] = []
    # URLs
    for m in re.finditer(r"https?://[^\s\"'<>]+", text or "", flags=re.I):
        try:
            host = urlparse(m.group(0)).hostname or ""
            if host:
                found.append(host.lower())
        except Exception:  # noqa: BLE001
            continue
    # bare domains
    for m in re.finditer(r"\b([a-z0-9-]+\.)+(com|cn|net|org|io|gov|edu)\b", text or "", flags=re.I):
        found.append(m.group(0).lower())
    return found


def check_window_title(title: str, cfg: SafetySection) -> TargetCheck:
    hit = _match_any(title, cfg.window_title_denylist)
    if hit:
        return TargetCheck(
            allowed=False,
            reason=f"窗口标题命中禁止列表: {hit}",
            blocked_by="window_title_denylist",
            matched=hit,
        )
    if cfg.window_title_allowlist:
        ok = _match_any(title, cfg.window_title_allowlist)
        if not ok:
            return TargetCheck(
                allowed=False,
                reason="窗口标题不在允许列表",
                blocked_by="window_title_allowlist",
            )
    return TargetCheck(allowed=True, reason="窗口标题通过")


def check_app_name(app: str, cfg: SafetySection) -> TargetCheck:
    hit = _match_any(app, cfg.app_denylist)
    if hit:
        return TargetCheck(
            allowed=False,
            reason=f"应用命中禁止列表: {hit}",
            blocked_by="app_denylist",
            matched=hit,
        )
    if cfg.app_allowlist:
        ok = _match_any(app, cfg.app_allowlist)
        if not ok:
            return TargetCheck(
                allowed=False,
                reason="应用不在允许列表",
                blocked_by="app_allowlist",
            )
    return TargetCheck(allowed=True, reason="应用通过")


def check_domains(texts: list[str], cfg: SafetySection) -> TargetCheck:
    domains: list[str] = []
    for t in texts:
        domains.extend(_extract_domains_from_text(t))
    for d in domains:
        hit = _match_any(d, cfg.domain_denylist)
        if hit:
            return TargetCheck(
                allowed=False,
                reason=f"域名命中禁止列表: {d}",
                blocked_by="domain_denylist",
                matched=d,
            )
    if cfg.domain_allowlist and domains:
        for d in domains:
            if not _match_any(d, cfg.domain_allowlist):
                return TargetCheck(
                    allowed=False,
                    reason=f"域名不在允许列表: {d}",
                    blocked_by="domain_allowlist",
                    matched=d,
                )
    return TargetCheck(allowed=True, reason="域名通过")


def _window_title_from_frame(frame: ScreenFrame | None) -> str:
    if frame is None:
        return ""
    win = getattr(frame, "window", None)
    if win is None:
        return ""
    return getattr(win, "title", "") or ""


def _app_from_frame(frame: ScreenFrame | None) -> str:
    if frame is None:
        return ""
    win = getattr(frame, "window", None)
    if win is None:
        return ""
    return getattr(win, "class_name", "") or getattr(win, "title", "") or ""


def _element_texts(vision: UIVisionResult | None, step: ActionStep) -> list[str]:
    texts: list[str] = [step.text or "", step.expected_change or ""]
    if vision is None:
        return texts
    el: UIElement | None = None
    if step.target_element_id:
        el = vision.by_id(step.target_element_id)
    if el is not None:
        texts.append(el.text or "")
        texts.append(el.role or "")
    # Sample top-level window-ish elements
    for e in vision.elements[:8]:
        texts.append(e.text or "")
    return texts


def check_targets(
    *,
    cfg: SafetySection,
    step: ActionStep,
    frame: ScreenFrame | None = None,
    vision: UIVisionResult | None = None,
    extra_text: str = "",
) -> TargetCheck:
    """Aggregate app / window / domain checks for one step."""
    title = _window_title_from_frame(frame)
    app = _app_from_frame(frame)
    # Also scan element text for denylisted window-like titles
    texts = _element_texts(vision, step)
    if extra_text:
        texts.append(extra_text)

    for t in texts:
        w = check_window_title(t, cfg)
        if not w.allowed and w.blocked_by == "window_title_denylist":
            return w

    if title:
        r = check_window_title(title, cfg)
        if not r.allowed:
            return r
    if app:
        r = check_app_name(app, cfg)
        if not r.allowed:
            return r

    r = check_domains(texts + [title, app], cfg)
    if not r.allowed:
        return r
    return TargetCheck(allowed=True, reason="目标通过允许/禁止列表")
