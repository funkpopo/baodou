"""Frame change detection via downsampled mean absolute difference."""

from __future__ import annotations

import hashlib

from PIL import Image, ImageChops, ImageFilter, ImageStat


class ChangeDetector:
    def __init__(self, *, sample_size: int = 64, threshold: float = 0.015) -> None:
        self.sample_size = max(8, sample_size)
        self.threshold = threshold
        self._prev: Image.Image | None = None
        self._prev_hash: str | None = None

    def reset(self) -> None:
        self._prev = None
        self._prev_hash = None

    def _sample(self, image: Image.Image) -> Image.Image:
        g = image.convert("L")
        return g.resize((self.sample_size, self.sample_size), Image.Resampling.BILINEAR)

    def fingerprint(self, image: Image.Image) -> str:
        sample = self._sample(image)
        return hashlib.blake2b(sample.tobytes(), digest_size=16).hexdigest()

    def score(self, image: Image.Image) -> float:
        """Return change score in roughly [0, 1]; 1.0 if no previous frame."""
        sample = self._sample(image)
        if self._prev is None:
            return 1.0
        diff = ImageChops.difference(sample, self._prev)
        # Slight blur reduces noise from cursor blink / video dither
        diff = diff.filter(ImageFilter.BoxBlur(1))
        stat = ImageStat.Stat(diff)
        # mean is 0..255
        return float(stat.mean[0]) / 255.0

    def update(self, image: Image.Image) -> tuple[bool, float, str]:
        """
        Compare against previous, then store current as baseline.
        Returns (changed, score, pixel_hash).
        """
        score = self.score(image)
        pixel_hash = self.fingerprint(image)
        changed = self._prev is None or score >= self.threshold or pixel_hash != self._prev_hash
        # Always advance baseline so gradual drift is tracked.
        self._prev = self._sample(image)
        self._prev_hash = pixel_hash
        return changed, score, pixel_hash

    def peek(self, image: Image.Image) -> tuple[bool, float, str]:
        """Score without updating baseline (for force paths that still want metrics)."""
        score = self.score(image)
        pixel_hash = self.fingerprint(image)
        changed = self._prev is None or score >= self.threshold
        return changed, score, pixel_hash
