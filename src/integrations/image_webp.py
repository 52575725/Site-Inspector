"""Auto-convert images to WebP format for better SEO performance."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_webp(image_path: str | Path, quality: int = 80) -> str | None:
    """Convert an image to WebP. Returns the new file path or None on failure."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed, cannot convert to WebP")
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    # Skip already-small images or already-WebP
    if path.suffix.lower() == ".webp":
        return str(path)
    if path.stat().st_size < 2048:  # skip tiny images
        return None

    try:
        img = Image.open(path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        webp_path = path.with_suffix(".webp")
        img.save(webp_path, "WEBP", quality=quality)

        original_size = path.stat().st_size
        new_size = webp_path.stat().st_size
        reduction = (1 - new_size / original_size) * 100 if original_size > 0 else 0
        logger.info(f"WebP: {path.name} {original_size//1024}KB → {new_size//1024}KB ({reduction:.0f}% smaller)")
        return str(webp_path)
    except Exception as e:
        logger.warning(f"WebP conversion failed for {path}: {e}")
        return None


def batch_convert_directory(directory: str | Path, quality: int = 80) -> dict:
    """Convert all images in a directory to WebP. Returns stats."""
    path = Path(directory)
    if not path.exists():
        return {"error": "Directory not found"}

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    results = {"converted": 0, "skipped": 0, "failed": 0, "saved_kb": 0}

    for img_path in path.rglob("*"):
        if img_path.suffix.lower() in supported:
            original_size = img_path.stat().st_size
            webp_path = convert_to_webp(img_path, quality)
            if webp_path:
                new_size = Path(webp_path).stat().st_size
                results["converted"] += 1
                results["saved_kb"] += (original_size - new_size) // 1024
            else:
                results["skipped"] += 1

    return results
