from __future__ import annotations

from pathlib import Path

from PIL import Image

DEMO_IMAGES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("color_wrap_grey_satin_installed.png", (160, 165, 168)),
    ("window_tint_black_privacy.png", (42, 48, 52)),
    ("ppf_clear_water_beading.png", (210, 220, 225)),
)


def ensure_demo_images(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    existing = [path for path in raw_dir.iterdir() if path.is_file()]
    if existing:
        return existing
    for filename, color in DEMO_IMAGES:
        path = raw_dir / filename
        Image.new("RGB", (640, 480), color=color).save(path)
        created.append(path)
    return created
