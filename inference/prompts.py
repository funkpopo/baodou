"""Versioned prompts for Qwen3.5-2B via llama-server.

Prompt changes must bump PROMPT_VERSION and add a regression fixture under
``benchmarks/phase_e/fixtures/`` so capability does not silently regress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Bump when system / task templates change incompatibly or alter expected behavior.
PROMPT_VERSION = "1.0.0"

# Allowed ActionType values the model may propose (whitelist for Phase E).
ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "none",
        "click",
        "double_click",
        "right_click",
        "move",
        "scroll",
        "type",
        "key",
        "hotkey",
        "wait",
        "reidentify",
    }
)

# High-risk action tokens that always require confirmation / may be blocked later.
SENSITIVE_ACTION_HINTS: frozenset[str] = frozenset(
    {"delete", "删除", "payment", "支付", "transfer", "转账", "password", "密码", "submit", "提交"}
)


@dataclass(frozen=True)
class PromptBundle:
    version: str
    name: str
    system: str
    stop: tuple[str, ...]
    description: str


# Stop sequences: Qwen chat ends assistant turns with these; keep empty if
# server/chat template already handles EOS. Extra stops reduce runaway text.
DEFAULT_STOP: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
)

_SYSTEM_OBSERVE = """You are baodou, a local Windows desktop assistant (read-first).
Reply with ONE JSON object only. No markdown fences, no thinking, no prose outside JSON.

Protocol version: 1.0.0
Prompt version: {prompt_version}

Rules:
1. Prefer element_id from the provided UI list. Do NOT invent bare screen coordinates when an id exists.
2. If unsure, lower confidence and set plan.steps to [] or action "none".
3. Never propose high-risk irreversible actions without requires_confirmation=true.
4. Allowed actions: none, click, double_click, right_click, move, scroll, type, key, hotkey, wait, reidentify.
5. Coordinates (if any) must be screen physical pixels and inside the frame bounds.
6. Keep observation concise (<= 400 chars). Max 8 ui_candidates. Max 3 plan steps.

JSON schema (all keys required at top level):
{{
  "kind": "observe_plan",
  "observation": string,
  "confidence": number (0..1),
  "notes": string,
  "ui_candidates": [
    {{"element_id": string, "type": string, "text": string, "confidence": number}}
  ],
  "plan": {{
    "goal": string,
    "steps": [
      {{
        "action": string,
        "target_element_id": string|null,
        "target_point": {{"x": int, "y": int}}|null,
        "text": string|null,
        "keys": [string],
        "risk": "low"|"medium"|"high",
        "requires_confirmation": boolean,
        "preconditions": [string],
        "expected_change": string
      }}
    ],
    "stop_if": [string]
  }},
  "needs_user_confirm": boolean
}}
"""

_SYSTEM_OBSERVE_ONLY = """You are baodou, a local Windows desktop screen observer.
Reply with ONE JSON object only. No markdown fences, no thinking.

Prompt version: {prompt_version}

Rules:
1. Describe the screen and reference element_id from the UI list when possible.
2. Do not propose executable actions other than none / wait / reidentify.
3. Keep observation concise.

JSON schema:
{{
  "kind": "observation",
  "observation": string,
  "confidence": number (0..1),
  "notes": string,
  "ui_candidates": [
    {{"element_id": string, "type": string, "text": string, "confidence": number}}
  ]
}}
"""


PROMPTS: dict[str, PromptBundle] = {
    "observe_plan": PromptBundle(
        version=PROMPT_VERSION,
        name="observe_plan",
        system=_SYSTEM_OBSERVE.format(prompt_version=PROMPT_VERSION),
        stop=DEFAULT_STOP,
        description="Screen observation + optional low-risk ActionPlan sketch",
    ),
    "observation": PromptBundle(
        version=PROMPT_VERSION,
        name="observation",
        system=_SYSTEM_OBSERVE_ONLY.format(prompt_version=PROMPT_VERSION),
        stop=DEFAULT_STOP,
        description="Read-only ScreenObservation JSON",
    ),
}


def get_prompt(name: str = "observe_plan") -> PromptBundle:
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt: {name}; known={list(PROMPTS)}")
    return PROMPTS[name]


def build_user_message(
    *,
    user_goal: str,
    frame_id: str,
    width: int,
    height: int,
    origin_x: int = 0,
    origin_y: int = 0,
    ui_summary: list[dict[str, Any]] | None = None,
    ui_text: str = "",
    mode: str = "observe_plan",
) -> str:
    """Assemble the text part of the user turn (image attached separately)."""
    lines = [
        f"User goal: {user_goal}",
        f"Frame: id={frame_id} image={width}x{height} origin=({origin_x},{origin_y}) "
        f"(coords are virtual-desktop physical pixels)",
        f"Mode: {mode}",
        "UI elements (prefer these element_ids):",
    ]
    if ui_text:
        lines.append(ui_text)
    elif ui_summary:
        for i, row in enumerate(ui_summary[:48], start=1):
            eid = row.get("element_id", f"e{i}")
            et = row.get("type", "?")
            text = row.get("text", "")
            click = row.get("clickable", False)
            conf = row.get("confidence", "")
            lines.append(f"  [{i}] id={eid} type={et} text={text!r} clickable={click} conf={conf}")
    else:
        lines.append("  (none)")
    lines.append(
        "Return the JSON object now. Prefer element_id over target_point. "
        "If the goal is read-only, use empty steps or action none."
    )
    return "\n".join(lines)


def prompt_registry_meta() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "prompts": {
            k: {
                "version": v.version,
                "description": v.description,
                "stop": list(v.stop),
                "system_chars": len(v.system),
            }
            for k, v in PROMPTS.items()
        },
        "allowed_actions": sorted(ALLOWED_ACTIONS),
    }
