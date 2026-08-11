"""HTTP client for D:\\llama llama-server (OpenAI-compatible). Phase B skeleton.

Does not start the server; health() and chat() are ready for Phase E wiring.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import AppConfig
from core.errors import ErrorCode, InferenceError
from core.logging import get_logger, log_event
from core.models import InferenceResponse, ScreenFrame, ScreenObservation, UIVisionResult

from inference.base import InferenceBackend

_log = get_logger("inference.http")


class HttpInference(InferenceBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base = config.inference.base_url

    def health(self) -> bool:
        url = f"{self.base}/health"
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=self.config.inference.health_timeout_sec) as resp:
                return 200 <= getattr(resp, "status", 200) < 300
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            log_event(_log, "inference.health_failed", error=str(exc), url=url)
            return False

    def observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
    ) -> InferenceResponse:
        """Minimal chat completion call — full prompt/schema belongs to Phase E."""
        if not self.health():
            raise InferenceError(
                "llama-server 不可用",
                code=ErrorCode.MODEL_UNAVAILABLE,
                details={"base_url": self.base},
            )

        ui_lines = [
            f"- {e.element_id} [{e.type.value}] text={e.text!r} clickable={e.clickable}"
            for e in vision.elements[:12]
        ]
        user_text = (
            f"User goal: {user_goal}\n"
            f"Frame: {frame.frame_id} {frame.width}x{frame.height}\n"
            f"UI elements:\n" + "\n".join(ui_lines) + "\n"
            "Reply with a single JSON object: "
            '{"observation": str, "notes": str, "confidence": number}'
        )
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a desktop assistant. Reply with one JSON object only. "
                        "No markdown fences."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            "max_tokens": self.config.inference.max_tokens,
            "temperature": self.config.inference.temperature,
            "chat_template_kwargs": {
                "enable_thinking": self.config.inference.enable_thinking,
            },
        }
        # Attach image if present (data URL).
        if frame.image_b64:
            body["messages"][-1]["content"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame.image_b64}"},
                },
                {"type": "text", "text": user_text},
            ]

        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.config.inference.request_timeout_sec) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"HTTP 推理失败: {exc}",
                code=ErrorCode.INFERENCE_FAILED,
                cause=exc,
            ) from exc

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                "响应缺少 choices/message/content",
                code=ErrorCode.OUTPUT_PARSE_FAILED,
                cause=exc,
                details={"raw_keys": list(raw) if isinstance(raw, dict) else type(raw).__name__},
            ) from exc

        observation = ScreenObservation(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            observation=str(content)[:2000],
            ui_elements=vision.elements[:6],
            notes="http raw content (Phase E will parse schema)",
            confidence=0.5,
            model_name="llama-server",
        )
        return InferenceResponse(
            request_id=f"http-{frame.frame_id}",
            trace_id=trace_id,
            ok=True,
            observation=observation,
            raw_text=str(content),
        )


def create_inference(config: AppConfig) -> InferenceBackend:
    if config.inference.backend == "http":
        return HttpInference(config)
    from inference.mock import MockInference

    return MockInference(config)
