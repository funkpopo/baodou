"""In-memory frame packet: metadata + PIL image (not logged as pixels)."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

from core.models import FrameKind, ScreenFrame
from PIL import Image


@dataclass
class FramePacket:
    meta: ScreenFrame
    image: Image.Image | None = None
    encoded: bytes | None = None
    kind: FrameKind = FrameKind.RAW
    _released: bool = field(default=False, repr=False)

    @property
    def frame_id(self) -> str:
        return self.meta.frame_id

    def release(self) -> None:
        """Drop pixel buffers to help GC under backpressure."""
        self.image = None
        self.encoded = None
        self._released = True

    def ensure_encoded(self, *, fmt: str | None = None, jpeg_quality: int = 85) -> bytes:
        if self.encoded is not None:
            return self.encoded
        if self.image is None:
            return b""
        from capture.preprocess import encode_image

        f = fmt or self.meta.image_format or "png"
        self.encoded = encode_image(self.image, fmt=f, jpeg_quality=jpeg_quality)
        self.meta.image_bytes = len(self.encoded)
        return self.encoded

    def attach_b64(self) -> str:
        data = self.ensure_encoded()
        b64 = base64.b64encode(data).decode("ascii")
        self.meta.image_b64 = b64
        return b64

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.image is not None:
            fmt = "JPEG" if self.meta.image_format.lower() in ("jpg", "jpeg") else "PNG"
            self.image.save(path, format=fmt)
        elif self.encoded is not None:
            path.write_bytes(self.encoded)
        else:
            raise ValueError("no image data to save")
        self.meta.image_path = str(path)
        return path
