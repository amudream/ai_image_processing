from __future__ import annotations

from pathlib import Path

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import sha256_file, stable_id
from app.core.states import ImageAssetStatus
from app.core.taxonomy import IMAGE_EXTENSIONS
from app.models import ImageAsset


class IngestionService:
    def __init__(self, db: Session, thumbnail_dir: Path = Path("data/thumbnails")) -> None:
        self.db = db
        self.thumbnail_dir = thumbnail_dir

    def import_folder(self, folder: Path, limit: int | None = None) -> list[ImageAsset]:
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {folder}")

        imported: list[ImageAsset] = []
        paths = [
            path
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for path in paths[:limit]:
            imported.append(self.import_file(path))
        self.db.commit()
        return imported

    def import_file(self, path: Path) -> ImageAsset:
        file_hash = sha256_file(path)
        existing = self.db.scalar(select(ImageAsset).where(ImageAsset.sha256 == file_hash))
        if existing is not None:
            return existing

        width, height = self._read_dimensions(path)
        asset = ImageAsset(
            id=stable_id("img", file_hash),
            source_uri=str(path.resolve()),
            sha256=file_hash,
            perceptual_hash=file_hash[:16],
            width=width,
            height=height,
            aspect_ratio=self._aspect_ratio(width, height),
            thumbnail_uri=self._thumbnail_placeholder(path),
            status=ImageAssetStatus.INGESTED.value,
        )
        self.db.add(asset)
        self.db.flush()
        return asset

    def _read_dimensions(self, path: Path) -> tuple[int | None, int | None]:
        try:
            with Image.open(path) as image:
                return image.size
        except OSError:
            return None, None

    def _thumbnail_placeholder(self, path: Path) -> str:
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        return str((self.thumbnail_dir / f"{path.stem}.thumb.txt").resolve())

    def _aspect_ratio(self, width: int | None, height: int | None) -> str | None:
        if not width or not height:
            return None
        if width == height:
            return "1:1"
        return f"{width}:{height}"
