from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops


def compute_screenshot_diff(before_path: str | Path, after_path: str | Path) -> float:
    """Compute pixel-level difference between two screenshots. Returns 0.0-1.0.

    0.0 = identical, 1.0 = completely different.
    """
    before = Image.open(before_path).convert("L")  # grayscale for faster comparison
    after = Image.open(after_path).convert("L")

    # Resize to same dimensions if needed
    if before.size != after.size:
        after = after.resize(before.size, Image.LANCZOS)

    diff = ImageChops.difference(before, after)
    diff_pixels = sum(1 for p in diff.getdata() if p > 10)  # threshold to ignore noise
    total_pixels = before.size[0] * before.size[1]
    return diff_pixels / total_pixels if total_pixels > 0 else 0.0
