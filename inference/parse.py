"""Extract and repair JSON from model raw text."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    """Parse first JSON object/array from model output.

    Returns (parsed, error_or_repair_note).
    """
    text = (text or "").strip()
    if not text:
        return None, "empty"

    # Strip common thinking / fence wrappers.
    text = _strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        val = json.loads(text)
        if isinstance(val, (dict, list)):
            return val, None
        return None, "json_root_not_object"
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None, "no_json_object"
    candidate = m.group(0)
    try:
        val = json.loads(candidate)
        if isinstance(val, (dict, list)):
            return val, None
        return None, "json_root_not_object"
    except json.JSONDecodeError as exc:
        repaired = try_repair_json(candidate)
        if repaired is not None:
            return repaired, f"repaired_after: {exc}"
        return None, f"json_error: {exc}"


def try_repair_json(s: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort close of truncated JSON (common when max_tokens hits)."""
    s = s.rstrip()
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s)

    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{" or ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    closers = {"{": "}", "[": "]"}
    while stack:
        s += closers[stack.pop()]
    try:
        val = json.loads(s)
        if isinstance(val, (dict, list)):
            return val
    except json.JSONDecodeError:
        return None
    return None


def is_json_complete(text: str) -> bool:
    """Heuristic: balanced braces/brackets and not inside a string — for stream gates."""
    s = (text or "").strip()
    if not s:
        return False
    s = _strip_thinking(s)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    # Find first object
    start = s.find("{")
    if start < 0:
        return False
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Trailing junk after complete object is ok for completeness of the object
                try:
                    json.loads(s[start : i + 1])
                    return True
                except json.JSONDecodeError:
                    return False
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return False


def _strip_thinking(text: str) -> str:
    # Qwen thinking blocks if accidentally enabled
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)
    return text.strip()
