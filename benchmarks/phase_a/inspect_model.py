"""Inspect local GGUF + mmproj metadata for Phase A."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gguf import GGUFReader
from gguf.constants import GGUFValueType

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
RESULTS = Path(__file__).resolve().parent / "results" / "model_metadata.json"

MODEL_PATH = MODEL_DIR / "Qwen3.5-2B-UD-Q4_K_XL.gguf"
MMPROJ_PATH = MODEL_DIR / "mmproj-F16.gguf"

KEY_WHITELIST = (
    "general.architecture",
    "general.type",
    "general.name",
    "general.basename",
    "general.size_label",
    "general.file_type",
    "general.quantized_by",
    "qwen35.block_count",
    "qwen35.context_length",
    "qwen35.embedding_length",
    "qwen35.feed_forward_length",
    "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv",
    "tokenizer.chat_template",
    "clip.has_vision_encoder",
    "clip.vision.projection_dim",
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.block_count",
    "clip.projector_type",
)


def _to_python(val):
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list) and len(val) == 1:
        val = val[0]
    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val).decode("utf-8", errors="replace")
    # Some gguf builds expose strings as list[int] codepoints/bytes
    if isinstance(val, list) and val and all(isinstance(x, int) for x in val):
        try:
            return bytes(val).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return val
    return val


def _decode_field(field) -> object:
    try:
        itype = field.types[0] if field.types else None
        if itype == GGUFValueType.STRING:
            raw = field.parts[field.data[0]]
            return _to_python(raw)
        if itype == GGUFValueType.ARRAY:
            return f"array(len={len(field.data)})"
        idx = field.data[0]
        val = field.parts[idx]
        return _to_python(val)
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def kv_summary(path: Path) -> dict:
    reader = GGUFReader(str(path))
    all_keys = list(reader.fields.keys())
    interesting: dict[str, object] = {}
    for key in all_keys:
        if key in KEY_WHITELIST or key.startswith("qwen35.") or key.startswith("clip."):
            if key.startswith("tokenizer.ggml.") and key not in ("tokenizer.chat_template",):
                continue
            interesting[key] = _decode_field(reader.fields[key])

    if "tokenizer.chat_template" in reader.fields:
        tmpl = _decode_field(reader.fields["tokenizer.chat_template"])
        if isinstance(tmpl, str):
            interesting["tokenizer.chat_template_preview"] = tmpl[:400]
            interesting["tokenizer.chat_template_len"] = len(tmpl)
            interesting.pop("tokenizer.chat_template", None)

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024**2), 2),
        "n_tensors": len(reader.tensors),
        "interesting_kv": interesting,
        "sample_tensor_names": [t.name for t in reader.tensors[:12]],
    }


def main() -> int:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "model": kv_summary(MODEL_PATH) if MODEL_PATH.exists() else {"error": "missing", "path": str(MODEL_PATH)},
        "mmproj": kv_summary(MMPROJ_PATH) if MMPROJ_PATH.exists() else {"error": "missing", "path": str(MMPROJ_PATH)},
        "vision_required_files": {
            "llm_gguf": MODEL_PATH.exists(),
            "mmproj_gguf": MMPROJ_PATH.exists(),
            "conclusion": (
                "Qwen3.5-2B GGUF is multimodal (image-text-to-text). "
                "llama.cpp needs matching mmproj-F16.gguf + MTMD / llama-server --mmproj for image input. "
                "Run on GPU via D:\\llama SYCL (Intel Arc), not CPU llama-cpp-python wheel."
                if MMPROJ_PATH.exists()
                else "mmproj missing — text-only only."
            ),
        },
        "practical_defaults": {
            "quant": "UD-Q4_K_XL (Unsloth Dynamic)",
            "native_context_tokens": 262144,
            "recommended_runtime_n_ctx_mvp": 4096,
            "n_gpu_layers": 99,
            "device": "SYCL0 (Intel Arc A770)",
            "backend": "D:\\llama\\llama-server.exe + oneAPI setvars",
            "vision_input": "Requires mmproj-F16.gguf + OpenAI-style image_url parts or --image",
            "do_not_use": "conda llama-cpp-python CPU wheel (ggml-cpu only) for production inference",
        },
    }
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
