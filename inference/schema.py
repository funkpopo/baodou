"""JSON Schema / GBNF grammar for constrained model output (Phase E).

llama-server (b10356+) accepts ``json_schema`` in the request body when supported.
We always keep a local Pydantic validation path as the source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from core.models import ActionType, RiskLevel
from pydantic import BaseModel, Field, field_validator

from inference.prompts import ALLOWED_ACTIONS

# ---------------------------------------------------------------------------
# Wire models (model output → validated before ActionPlan / ScreenObservation)
# ---------------------------------------------------------------------------


class ModelUICandidate(BaseModel):
    element_id: str = ""
    type: str = "other"
    text: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelPoint(BaseModel):
    x: int
    y: int


class ModelActionStep(BaseModel):
    action: str = "none"
    target_element_id: str | None = None
    target_point: ModelPoint | None = None
    text: str | None = None
    keys: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "low"
    requires_confirmation: bool = True
    preconditions: list[str] = Field(default_factory=list)
    expected_change: str = ""

    @field_validator("action")
    @classmethod
    def _action_known(cls, v: str) -> str:
        a = (v or "none").strip().lower()
        if a not in ALLOWED_ACTIONS:
            # Normalize unknowns to none — validate layer may still reject.
            return a
        return a


class ModelActionPlan(BaseModel):
    goal: str = ""
    steps: list[ModelActionStep] = Field(default_factory=list)
    stop_if: list[str] = Field(default_factory=list)


class ModelObservePlan(BaseModel):
    """Primary structured output for observe+plan requests."""

    kind: str = "observe_plan"
    observation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""
    ui_candidates: list[ModelUICandidate] = Field(default_factory=list)
    plan: ModelActionPlan = Field(default_factory=ModelActionPlan)
    needs_user_confirm: bool = True


class ModelObservationOnly(BaseModel):
    kind: str = "observation"
    observation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""
    ui_candidates: list[ModelUICandidate] = Field(default_factory=list)


# OpenAPI-ish JSON Schema for llama-server ``json_schema`` parameter.
OBSERVE_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string"},
        "observation": {"type": "string"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
        "ui_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "element_id": {"type": "string"},
                    "type": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["element_id", "type", "text", "confidence"],
            },
        },
        "plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "goal": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": sorted(ALLOWED_ACTIONS),
                            },
                            "target_element_id": {"type": ["string", "null"]},
                            "target_point": {
                                "type": ["object", "null"],
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                },
                                "required": ["x", "y"],
                            },
                            "text": {"type": ["string", "null"]},
                            "keys": {"type": "array", "items": {"type": "string"}},
                            "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                            "requires_confirmation": {"type": "boolean"},
                            "preconditions": {"type": "array", "items": {"type": "string"}},
                            "expected_change": {"type": "string"},
                        },
                        "required": [
                            "action",
                            "risk",
                            "requires_confirmation",
                        ],
                    },
                },
                "stop_if": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["goal", "steps"],
        },
        "needs_user_confirm": {"type": "boolean"},
    },
    "required": [
        "kind",
        "observation",
        "confidence",
        "notes",
        "ui_candidates",
        "plan",
        "needs_user_confirm",
    ],
}


OBSERVATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string"},
        "observation": {"type": "string"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
        "ui_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "element_id": {"type": "string"},
                    "type": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["element_id", "type", "text", "confidence"],
            },
        },
    },
    "required": ["kind", "observation", "confidence", "notes", "ui_candidates"],
}


# Compact GBNF for servers that prefer grammar over json_schema.
# Intentionally permissive on strings; strict checks happen in validate.py.
OBSERVE_PLAN_GBNF = r"""
root ::= "{" ws kind-kv "," ws observation-kv "," ws confidence-kv "," ws notes-kv "," ws candidates-kv "," ws plan-kv "," ws confirm-kv ws "}"
kind-kv ::= "\"kind\"" ws ":" ws string
observation-kv ::= "\"observation\"" ws ":" ws string
confidence-kv ::= "\"confidence\"" ws ":" ws number
notes-kv ::= "\"notes\"" ws ":" ws string
candidates-kv ::= "\"ui_candidates\"" ws ":" ws "[" ws (candidate (ws "," ws candidate)*)? ws "]"
candidate ::= "{" ws "\"element_id\"" ws ":" ws string "," ws "\"type\"" ws ":" ws string "," ws "\"text\"" ws ":" ws string "," ws "\"confidence\"" ws ":" ws number ws "}"
plan-kv ::= "\"plan\"" ws ":" ws plan-obj
plan-obj ::= "{" ws "\"goal\"" ws ":" ws string "," ws "\"steps\"" ws ":" ws "[" ws (step (ws "," ws step)*)? ws "]" ("," ws "\"stop_if\"" ws ":" ws string-array)? ws "}"
step ::= "{" ws "\"action\"" ws ":" ws action-enum "," ws "\"risk\"" ws ":" ws risk-enum "," ws "\"requires_confirmation\"" ws ":" ws boolean ("," ws "\"target_element_id\"" ws ":" ws (string | null))? ("," ws "\"expected_change\"" ws ":" ws string)? ws "}"
confirm-kv ::= "\"needs_user_confirm\"" ws ":" ws boolean
action-enum ::= "\"none\"" | "\"click\"" | "\"double_click\"" | "\"right_click\"" | "\"move\"" | "\"scroll\"" | "\"type\"" | "\"key\"" | "\"hotkey\"" | "\"wait\"" | "\"reidentify\""
risk-enum ::= "\"low\"" | "\"medium\"" | "\"high\""
string-array ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \t\n\r]*
"""


def schema_for_mode(mode: str) -> dict[str, Any]:
    if mode == "observation":
        return OBSERVATION_JSON_SCHEMA
    return OBSERVE_PLAN_JSON_SCHEMA


def parse_model_payload(
    data: dict[str, Any], *, mode: str = "observe_plan"
) -> ModelObservePlan | ModelObservationOnly:
    if mode == "observation" or data.get("kind") == "observation":
        return ModelObservationOnly.model_validate(_normalize_observation_dict(data))
    return ModelObservePlan.model_validate(_normalize_observe_plan_dict(data))


def _normalize_observation_dict(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.setdefault("kind", "observation")
    out.setdefault("observation", str(data.get("observation") or data.get("summary") or ""))
    out.setdefault("confidence", float(data.get("confidence") or 0.5))
    out.setdefault("notes", str(data.get("notes") or ""))
    out.setdefault("ui_candidates", data.get("ui_candidates") or data.get("ui_elements") or [])
    # Map alternate shapes from Phase A free-form
    if out["ui_candidates"] and isinstance(out["ui_candidates"], list):
        fixed = []
        for c in out["ui_candidates"]:
            if not isinstance(c, dict):
                continue
            fixed.append(
                {
                    "element_id": str(c.get("element_id") or c.get("id") or ""),
                    "type": str(c.get("type") or "other"),
                    "text": str(c.get("text") or ""),
                    "confidence": float(c.get("confidence") or 0.5),
                }
            )
        out["ui_candidates"] = fixed
    return out


def _normalize_observe_plan_dict(data: dict[str, Any]) -> dict[str, Any]:
    out = _normalize_observation_dict(data)
    out["kind"] = str(data.get("kind") or "observe_plan")
    plan = data.get("plan")
    if not isinstance(plan, dict):
        # Lift suggested_action from Phase A shape
        sug = data.get("suggested_action")
        if isinstance(sug, dict):
            plan = {
                "goal": str(data.get("goal") or ""),
                "steps": [
                    {
                        "action": sug.get("action") or "none",
                        "target_element_id": sug.get("target_element_id"),
                        "risk": sug.get("risk") or "low",
                        "requires_confirmation": bool(sug.get("requires_confirmation", True)),
                        "expected_change": str(sug.get("expected_change") or ""),
                    }
                ],
                "stop_if": ["target_missing", "confidence_below_threshold"],
            }
        else:
            plan = {"goal": str(data.get("goal") or ""), "steps": [], "stop_if": []}
    out["plan"] = plan
    out.setdefault("needs_user_confirm", True)
    # If any step needs confirmation, force flag
    steps = plan.get("steps") if isinstance(plan, dict) else []
    if isinstance(steps, list) and any(
        isinstance(s, dict)
        and s.get("requires_confirmation", True)
        and s.get("action", "none") != "none"
        for s in steps
    ):
        out["needs_user_confirm"] = True
    return out


def action_type_or_none(raw: str) -> ActionType:
    try:
        return ActionType(raw)
    except ValueError:
        return ActionType.NONE


def risk_or_low(raw: str) -> RiskLevel:
    try:
        return RiskLevel(raw)
    except ValueError:
        return RiskLevel.LOW
