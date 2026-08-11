"""HTTP client for D:\\llama llama-server (OpenAI-compatible) — Phase E.

Features:
  - lifecycle via LlamaServerManager (load / warmup / recover)
  - multimodal messages (UI summary + optional image)
  - prompt templates + json_schema / grammar constraints
  - stream with gate: only complete validated JSON may enter action layer
  - schema / element / coord / whitelist validation
  - timeout, retry, cancel token, degrade on failure
"""

from __future__ import annotations

import base64
import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.cancel import get_global_token
from core.config import AppConfig
from core.errors import ErrorCode, InferenceError
from core.logging import get_logger, log_event
from core.models import (
    InferenceRequest,
    InferenceResponse,
    ScreenFrame,
    ScreenObservation,
    UIVisionResult,
)
from ui_vision.context import serialize_for_model, serialize_text_summary

from inference.base import InferenceBackend
from inference.degrade import degraded_response
from inference.parse import extract_json, is_json_complete
from inference.prompts import PROMPT_VERSION, build_user_message, get_prompt
from inference.schema import schema_for_mode
from inference.server import LlamaServerManager
from inference.validate import validate_model_output

_log = get_logger("inference.http")


class HttpInference(InferenceBackend):
    def __init__(
        self,
        config: AppConfig,
        *,
        server: LlamaServerManager | None = None,
        last_observation: ScreenObservation | None = None,
    ) -> None:
        self.config = config
        self.base = config.inference.base_url
        self.server = server or LlamaServerManager(config)
        self._last_observation = last_observation
        self._cancel_flag = False
        self._prompt_name = "observe_plan"

    # ------------------------------------------------------------------ health / lifecycle
    def health(self) -> bool:
        return self.server.health()

    def ensure_ready(self) -> dict[str, Any]:
        return self.server.ensure_running(warmup=self.config.inference.warmup_on_start)

    def cancel_current(self) -> None:
        self._cancel_flag = True

    def close(self) -> None:
        # Do not stop externally shared server by default — only managed one if we own it.
        # Explicit stop via CLI `infer server stop`.
        return None

    def set_last_observation(self, obs: ScreenObservation | None) -> None:
        self._last_observation = obs

    # ------------------------------------------------------------------ public API
    def observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
        mode: str = "observe_plan",
        include_image: bool = True,
    ) -> InferenceResponse:
        get_global_token().check()
        self._cancel_flag = False
        t0 = time.perf_counter()
        req = InferenceRequest(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            task=user_goal,
            system_prompt=get_prompt(
                mode if mode in ("observe_plan", "observation") else "observe_plan"
            ).system[:200],
            ui_summary=serialize_for_model(
                vision, goal=user_goal, max_elements=self.config.ui_vision.context_max_elements
            ),
            has_image=bool(include_image and (frame.image_b64 or frame.image_path)),
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
        )
        log_event(
            _log,
            "inference.request",
            request_id=req.request_id,
            trace_id=trace_id,
            frame_id=frame.frame_id,
            task_preview=user_goal[:80],
            element_count=len(vision.elements),
            mode=mode,
            prompt_version=PROMPT_VERSION,
            has_image=req.has_image,
        )

        try:
            if not self.health():
                if self.config.inference.auto_start_server:
                    self.ensure_ready()
                if not self.health():
                    raise InferenceError(
                        "llama-server 不可用",
                        code=ErrorCode.MODEL_UNAVAILABLE,
                        details={"base_url": self.base},
                    )

            body = self._build_chat_body(
                frame=frame,
                vision=vision,
                user_goal=user_goal,
                mode=mode,
                include_image=include_image,
            )

            raw_text, meta = self._chat_with_retries(body, request_id=req.request_id)
            latency_ms = (time.perf_counter() - t0) * 1000
            finish = meta.get("finish_reason")
            truncated = finish == "length"

            parsed, parse_note = extract_json(raw_text)
            if parsed is None:
                raise InferenceError(
                    f"输出解析失败: {parse_note}",
                    code=ErrorCode.OUTPUT_PARSE_FAILED,
                    details={"parse_note": parse_note, "raw_preview": raw_text[:400]},
                )

            v = validate_model_output(
                parsed if isinstance(parsed, dict) else None,
                frame=frame,
                vision=vision,
                user_goal=user_goal,
                mode=mode,
                trace_id=trace_id,
                model_name=meta.get("model") or "llama-server",
                latency_ms=latency_ms,
                raw_truncated=truncated,
            )

            if not v.ok or v.observation is None:
                # Degrade rather than leak invalid plan
                reason = "; ".join(v.fatal_messages() or [i.message for i in v.issues]) or "invalid"
                log_event(
                    _log,
                    "inference.validate_failed",
                    request_id=req.request_id,
                    reason=reason,
                    parse_note=parse_note,
                    issues=[{"code": i.code, "message": i.message} for i in v.issues],
                )
                if self.config.inference.degrade_on_error:
                    resp = degraded_response(
                        request_id=req.request_id,
                        trace_id=trace_id,
                        frame=frame,
                        vision=vision,
                        user_goal=user_goal,
                        reason=reason,
                        error_code=ErrorCode.OUTPUT_SCHEMA_INVALID.value,
                        last_observation=self._last_observation,
                        latency_ms=latency_ms,
                        raw_text=raw_text,
                    )
                    # Attach observation if schema partially ok
                    if v.observation is not None:
                        resp.observation = v.observation
                        resp.observation.notes = (resp.observation.notes + f" | invalid: {reason}")[
                            :1000
                        ]
                    log_event(_log, "inference.response", **resp.log_summary(), degraded=True)
                    return resp
                raise InferenceError(
                    reason,
                    code=ErrorCode.OUTPUT_SCHEMA_INVALID,
                    details={
                        "issues": [
                            {"code": i.code, "message": i.message, "path": i.path} for i in v.issues
                        ]
                    },
                )

            if v.observation:
                self._last_observation = v.observation

            resp = InferenceResponse(
                request_id=req.request_id,
                trace_id=trace_id,
                ok=True,
                observation=v.observation,
                plan=v.plan,
                raw_text=raw_text,
                latency_ms=latency_ms,
            )
            log_event(
                _log,
                "inference.response",
                **resp.log_summary(),
                parse_note=parse_note,
                issue_count=len(v.issues),
                prompt_version=PROMPT_VERSION,
            )
            return resp

        except InferenceError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            if self.config.inference.degrade_on_error and exc.code in {
                ErrorCode.MODEL_UNAVAILABLE,
                ErrorCode.INFERENCE_TIMEOUT,
                ErrorCode.INFERENCE_FAILED,
                ErrorCode.OUTPUT_PARSE_FAILED,
                ErrorCode.OUTPUT_SCHEMA_INVALID,
            }:
                resp = degraded_response(
                    request_id=req.request_id,
                    trace_id=trace_id,
                    frame=frame,
                    vision=vision,
                    user_goal=user_goal,
                    reason=exc.message,
                    error_code=exc.code.value,
                    last_observation=self._last_observation,
                    latency_ms=latency_ms,
                )
                log_event(_log, "inference.degraded", **resp.log_summary())
                return resp
            raise
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000
            if self.config.inference.degrade_on_error:
                resp = degraded_response(
                    request_id=req.request_id,
                    trace_id=trace_id,
                    frame=frame,
                    vision=vision,
                    user_goal=user_goal,
                    reason=str(exc),
                    error_code=ErrorCode.INFERENCE_FAILED.value,
                    last_observation=self._last_observation,
                    latency_ms=latency_ms,
                )
                log_event(
                    _log, "inference.degraded", **resp.log_summary(), cause=type(exc).__name__
                )
                return resp
            raise InferenceError(
                f"HTTP 推理失败: {exc}",
                code=ErrorCode.INFERENCE_FAILED,
                cause=exc,
            ) from exc

    def stream_observe(
        self,
        frame: ScreenFrame,
        vision: UIVisionResult,
        *,
        user_goal: str,
        trace_id: str = "",
        mode: str = "observe_plan",
        include_image: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Stream tokens; only emit ready_for_action after full validated JSON."""
        get_global_token().check()
        self._cancel_flag = False
        t0 = time.perf_counter()
        req = InferenceRequest(
            trace_id=trace_id,
            frame_id=frame.frame_id,
            task=user_goal,
            has_image=bool(include_image and (frame.image_b64 or frame.image_path)),
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
        )

        if not self.health():
            if self.config.inference.auto_start_server:
                self.ensure_ready()
            if not self.health():
                yield {
                    "type": "error",
                    "ready_for_action": False,
                    "error": "model_unavailable",
                }
                return

        body = self._build_chat_body(
            frame=frame,
            vision=vision,
            user_goal=user_goal,
            mode=mode,
            include_image=include_image,
        )
        body["stream"] = True

        accumulated = ""
        try:
            for piece in self._iter_sse_chat(body):
                get_global_token().check()
                if self._cancel_flag:
                    yield {"type": "cancelled", "ready_for_action": False, "raw_text": accumulated}
                    return
                accumulated += piece
                complete = is_json_complete(accumulated)
                yield {
                    "type": "delta",
                    "ready_for_action": False,
                    "delta": piece,
                    "raw_text": accumulated,
                    "json_complete": complete,
                }
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "error",
                "ready_for_action": False,
                "error": str(exc),
                "raw_text": accumulated,
            }
            return

        latency_ms = (time.perf_counter() - t0) * 1000
        if not is_json_complete(accumulated):
            yield {
                "type": "final",
                "ready_for_action": False,
                "error": "incomplete_json",
                "raw_text": accumulated,
                "latency_ms": latency_ms,
            }
            return

        parsed, parse_note = extract_json(accumulated)
        v = validate_model_output(
            parsed if isinstance(parsed, dict) else None,
            frame=frame,
            vision=vision,
            user_goal=user_goal,
            mode=mode,
            trace_id=trace_id,
            model_name="llama-server",
            latency_ms=latency_ms,
        )
        if not v.ok or v.observation is None:
            yield {
                "type": "final",
                "ready_for_action": False,
                "error": "schema_invalid",
                "parse_note": parse_note,
                "issues": [{"code": i.code, "message": i.message} for i in v.issues],
                "raw_text": accumulated,
                "latency_ms": latency_ms,
            }
            return

        self._last_observation = v.observation
        resp = InferenceResponse(
            request_id=req.request_id,
            trace_id=trace_id,
            ok=True,
            observation=v.observation,
            plan=v.plan,
            raw_text=accumulated,
            latency_ms=latency_ms,
        )
        # Gate: plan may exist but action layer still needs safety (Phase F).
        # We only mark ready_for_action when JSON is complete AND validated.
        yield {
            "type": "final",
            "ready_for_action": True,
            "response": resp,
            "raw_text": accumulated,
            "parse_note": parse_note,
            "latency_ms": latency_ms,
        }

    def chat(
        self,
        *,
        user_text: str,
        system: str | None = None,
        image_b64: str | None = None,
        max_tokens: int | None = None,
        trace_id: str = "",
    ) -> dict[str, Any]:
        """Raw chat helper for benches / diagnostics."""
        prompt = get_prompt("observation")
        content: Any
        if image_b64:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": user_text},
            ]
        else:
            content = user_text
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system or prompt.system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens or self.config.inference.max_tokens,
            "temperature": self.config.inference.temperature,
            "chat_template_kwargs": {"enable_thinking": self.config.inference.enable_thinking},
        }
        if prompt.stop:
            body["stop"] = list(prompt.stop)
        text, meta = self._chat_with_retries(body, request_id=f"chat-{trace_id or 'x'}")
        return {"text": text, "meta": meta}

    # ------------------------------------------------------------------ internals
    def _build_chat_body(
        self,
        *,
        frame: ScreenFrame,
        vision: UIVisionResult,
        user_goal: str,
        mode: str,
        include_image: bool,
    ) -> dict[str, Any]:
        prompt_name = mode if mode in ("observe_plan", "observation") else "observe_plan"
        prompt = get_prompt(prompt_name)
        ui_rows = serialize_for_model(
            vision, goal=user_goal, max_elements=self.config.ui_vision.context_max_elements
        )
        ui_text = serialize_text_summary(
            vision, goal=user_goal, max_elements=self.config.ui_vision.context_max_elements
        )
        user_text = build_user_message(
            user_goal=user_goal,
            frame_id=frame.frame_id,
            width=frame.width,
            height=frame.height,
            origin_x=frame.origin_x,
            origin_y=frame.origin_y,
            ui_summary=ui_rows,
            ui_text=ui_text,
            mode=prompt_name,
        )

        image_b64 = None
        if include_image:
            image_b64 = frame.image_b64
            if not image_b64 and frame.image_path:
                p = Path(frame.image_path)
                if p.exists():
                    image_b64 = base64.b64encode(p.read_bytes()).decode("ascii")

        if image_b64:
            user_content: Any = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": user_text},
            ]
        else:
            user_content = user_text

        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.config.inference.max_tokens,
            "temperature": self.config.inference.temperature,
            "top_p": 0.9,
            "chat_template_kwargs": {
                "enable_thinking": self.config.inference.enable_thinking,
            },
        }
        if prompt.stop:
            body["stop"] = list(prompt.stop)

        # Constraint: prefer json_schema; grammar as alternative
        cmode = self.config.inference.constraint_mode
        if cmode == "json_schema":
            body["json_schema"] = schema_for_mode(prompt_name)
        elif cmode == "grammar":
            from inference.schema import OBSERVE_PLAN_GBNF

            body["grammar"] = OBSERVE_PLAN_GBNF
        # "none" → prompt-only (still validated client-side)

        return body

    def _chat_with_retries(
        self, body: dict[str, Any], *, request_id: str
    ) -> tuple[str, dict[str, Any]]:
        retries = max(0, self.config.inference.max_retries)
        last_exc: BaseException | None = None
        for attempt in range(retries + 1):
            get_global_token().check()
            if self._cancel_flag:
                raise InferenceError("推理已取消", code=ErrorCode.CANCELLED)
            try:
                return self._chat_once(body)
            except InferenceError as exc:
                last_exc = exc
                retriable = exc.code in {
                    ErrorCode.INFERENCE_TIMEOUT,
                    ErrorCode.INFERENCE_FAILED,
                    ErrorCode.MODEL_UNAVAILABLE,
                }
                log_event(
                    _log,
                    "inference.retry",
                    request_id=request_id,
                    attempt=attempt,
                    code=exc.code.value,
                    retriable=retriable,
                )
                if not retriable or attempt >= retries:
                    raise
                # Brief backoff; try recover once on unavailable
                if (
                    exc.code == ErrorCode.MODEL_UNAVAILABLE
                    and self.config.inference.auto_start_server
                ):
                    with contextlib.suppress(Exception):
                        self.server.recover()
                time.sleep(0.4 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= retries:
                    raise InferenceError(
                        f"HTTP 推理失败: {exc}",
                        code=ErrorCode.INFERENCE_FAILED,
                        cause=exc,
                    ) from exc
                time.sleep(0.4 * (attempt + 1))
        raise InferenceError(
            f"HTTP 推理失败: {last_exc}",
            code=ErrorCode.INFERENCE_FAILED,
            cause=last_exc,
        )

    def _chat_once(self, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        timeout = self.config.inference.request_timeout_sec
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            raise InferenceError(
                "模型推理超时",
                code=ErrorCode.INFERENCE_TIMEOUT,
                cause=exc,
            ) from exc
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                detail = str(exc)
            # Some builds reject json_schema — retry once without constraint
            if exc.code in {400, 422} and ("json_schema" in body or "grammar" in body):
                body2 = dict(body)
                body2.pop("json_schema", None)
                body2.pop("grammar", None)
                log_event(
                    _log, "inference.constraint_fallback", http_code=exc.code, detail=detail[:200]
                )
                return self._chat_once(body2)
            raise InferenceError(
                f"HTTP 推理失败: {exc.code} {detail}",
                code=ErrorCode.INFERENCE_FAILED,
                cause=exc,
                details={"http_code": exc.code},
            ) from exc
        except (URLError, OSError) as exc:
            raise InferenceError(
                f"llama-server 连接失败: {exc}",
                code=ErrorCode.MODEL_UNAVAILABLE,
                cause=exc,
            ) from exc

        try:
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                "响应缺少 choices/message/content",
                code=ErrorCode.OUTPUT_PARSE_FAILED,
                cause=exc,
                details={"raw_keys": list(raw) if isinstance(raw, dict) else type(raw).__name__},
            ) from exc

        meta = {
            "finish_reason": finish,
            "model": raw.get("model"),
            "usage": raw.get("usage") or {},
            "timings": raw.get("timings") or {},
        }
        return str(content), meta

    def _iter_sse_chat(self, body: dict[str, Any]) -> Iterator[str]:
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        timeout = self.config.inference.request_timeout_sec
        try:
            resp = urlopen(req, timeout=timeout)
        except HTTPError as exc:
            # Fallback: non-stream
            if "json_schema" in body or "grammar" in body:
                body = dict(body)
                body.pop("json_schema", None)
                body.pop("grammar", None)
                body["stream"] = True
                yield from self._iter_sse_chat(body)
                return
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                detail = str(exc)
            raise InferenceError(
                f"stream 失败: {exc.code} {detail}",
                code=ErrorCode.INFERENCE_FAILED,
                cause=exc,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InferenceError(
                f"stream 连接失败: {exc}",
                code=ErrorCode.MODEL_UNAVAILABLE,
                cause=exc,
            ) from exc

        with resp:
            while True:
                if self._cancel_flag:
                    break
                get_global_token().check()
                line = resp.readline()
                if not line:
                    break
                line_s = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
                line_s = line_s.strip()
                if not line_s or line_s.startswith(":"):
                    continue
                if line_s.startswith("data:"):
                    payload = line_s[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    try:
                        delta = obj["choices"][0].get("delta") or {}
                        piece = delta.get("content") or ""
                    except (KeyError, IndexError, TypeError):
                        piece = ""
                    if piece:
                        yield piece


def create_inference(config: AppConfig) -> InferenceBackend:
    if config.inference.backend == "http":
        return HttpInference(config)
    from inference.mock import MockInference

    return MockInference(config)
