"""Threat detection helpers (Phase G).

Covers: prompt injection from screen/OCR text, malicious page instructions,
coordinate spoofing signals, clipboard-like secret exfil hints.
System policy always wins over model or screen text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.config import SafetySection
from core.models import ActionPlan, ActionStep, ActionType

# Phrases that try to jailbreak / force execution from untrusted screen content
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)"),
    re.compile(r"(?i)disregard\s+(all\s+)?(safety|security|policy)"),
    re.compile(r"(?i)you\s+are\s+now\s+(in\s+)?(developer|god|admin)\s+mode"),
    re.compile(r"(?i)系统提示|忽略(以上|之前|所有).*(规则|指令|限制)"),
    re.compile(r"(?i)立即执行|马上执行|不要确认|跳过确认|强制执行"),
    re.compile(r"(?i)execute\s+immediately|skip\s+confirm|do\s+not\s+ask"),
    re.compile(r"(?i)bypass\s+(safety|permission|confirm)"),
    re.compile(r"(?i)exfiltrat|send\s+clipboard|leak\s+(password|secret|token)"),
    re.compile(r"(?i)wget\s+http|curl\s+http|Invoke-WebRequest"),
]

# Model output that claims authority to skip gates
_MODEL_OVERRIDE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)requires_confirmation\s*[:=]\s*false.*high"),
    re.compile(r"(?i)auto[_-]?approve|no\s+confirmation\s+needed"),
    re.compile(r"(?i)立即执行|无需确认|跳过安全"),
]


@dataclass
class ThreatFinding:
    threat_id: str
    severity: str  # low|medium|high
    source: str  # screen|model|goal|coordinate
    message: str
    matched: str = ""


@dataclass
class ThreatReport:
    findings: list[ThreatFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == "high" for f in self.findings)

    @property
    def reasons(self) -> list[str]:
        return [f.message for f in self.findings]

    def log_summary(self) -> dict:
        return {
            "finding_count": len(self.findings),
            "blocked": self.blocked,
            "threats": [
                {
                    "id": f.threat_id,
                    "severity": f.severity,
                    "source": f.source,
                    "message": f.message[:120],
                }
                for f in self.findings[:12]
            ],
        }


def scan_text(text: str, *, source: str, cfg: SafetySection) -> list[ThreatFinding]:
    if not text or not cfg.block_prompt_injection:
        return []
    out: list[ThreatFinding] = []
    for i, pat in enumerate(_INJECTION_PATTERNS):
        m = pat.search(text)
        if m:
            out.append(
                ThreatFinding(
                    threat_id=f"injection_{i}",
                    severity="high",
                    source=source,
                    message="检测到提示注入/越权执行指令（不可信屏幕或文本）",
                    matched=m.group(0)[:80],
                )
            )
    for i, pat in enumerate(_MODEL_OVERRIDE_PATTERNS):
        m = pat.search(text)
        if m and source in ("model", "goal"):
            out.append(
                ThreatFinding(
                    threat_id=f"override_{i}",
                    severity="medium",
                    source=source,
                    message="检测到试图绕过确认的表述（策略仍优先生效）",
                    matched=m.group(0)[:80],
                )
            )
    return out


def scan_coordinate_spoof(step: ActionStep) -> list[ThreatFinding]:
    """Flag bare-coordinate high-impact actions without element_id."""
    out: list[ThreatFinding] = []
    if step.target_element_id:
        return out
    if step.target_point is None:
        return out
    if step.action in (
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.RIGHT_CLICK,
        ActionType.DRAG,
        ActionType.TYPE,
    ):
        if not step.allow_coordinate_fallback:
            out.append(
                ThreatFinding(
                    threat_id="coord_no_element",
                    severity="medium",
                    source="coordinate",
                    message="裸坐标动作且未声明 allow_coordinate_fallback（坐标欺骗风险）",
                )
            )
        # Off-screen absurd coordinates
        if step.target_point.x < -10000 or step.target_point.y < -10000:
            out.append(
                ThreatFinding(
                    threat_id="coord_absurd",
                    severity="high",
                    source="coordinate",
                    message="坐标异常越界，疑似欺骗",
                )
            )
    return out


def scan_plan(
    plan: ActionPlan,
    *,
    cfg: SafetySection,
    screen_texts: list[str] | None = None,
) -> ThreatReport:
    report = ThreatReport()
    # User goal is trusted relative to screen, but still scan for override language
    report.findings.extend(scan_text(plan.goal, source="goal", cfg=cfg))
    if cfg.ignore_screen_instructions:
        for t in screen_texts or []:
            # Screen text injection is untrusted — high severity findings block
            report.findings.extend(scan_text(t, source="screen", cfg=cfg))
    for step in plan.steps:
        blob = " ".join(
            [
                step.text or "",
                step.description or "",
                step.expected_change or "",
                " ".join(step.keys or []),
            ]
        )
        report.findings.extend(scan_text(blob, source="model", cfg=cfg))
        report.findings.extend(scan_coordinate_spoof(step))
    return report


# Static threat model summary for docs / CLI
THREAT_MODEL = [
    {
        "id": "prompt_injection",
        "title": "屏幕/OCR 提示注入",
        "mitigation": "ignore_screen_instructions + block_prompt_injection；策略优先于模型",
    },
    {
        "id": "malicious_page",
        "title": "恶意网页指令",
        "mitigation": "屏幕文字视为不可信；高风险关键词硬拦截；默认 read_only",
    },
    {
        "id": "model_hallucination",
        "title": "模型幻觉动作/坐标",
        "mitigation": "schema 校验、element_id 优先、执行前重定位、失败暂停",
    },
    {
        "id": "coordinate_spoof",
        "title": "坐标欺骗",
        "mitigation": "禁止无 element_id 裸坐标除非二次确认；边界与移动范围限制",
    },
    {
        "id": "clipboard_leak",
        "title": "剪贴板/密钥泄漏",
        "mitigation": "PII 脱敏、密码框遮罩、审计不写 raw 像素与 secret 字段",
    },
    {
        "id": "privilege_abuse",
        "title": "越权操作",
        "mitigation": "动作白名单、应用/窗口禁止列表、紧急停止、频率与时长限制",
    },
]
