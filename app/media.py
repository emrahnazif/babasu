"""Media files: stored once in MEDIA_DIR, named by the SHA-256 of their content.

Each image also gets a small WebP thumbnail next to it (<sha256>.thumb.webp) for grids.
"""

import hashlib
import io
import json
import sqlite3
from pathlib import Path

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from app import config

# The supported image formats: Pillow format name -> extension of the stored file.
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif"}
SUFFIXES = {f".{ext}" for ext in EXTENSIONS.values()} | {".jpeg", ".tiff"}  # file names to look for
FORMATS_TEXT = "JPEG, PNG, WebP, BMP and TIFF"  # for people
THUMB_SIZE = 256  # longest side, in pixels


class NotAnImage(ValueError):
    """The data isn't an image in one of the supported formats."""


def _relpath(sha256: str, ext: str) -> str:
    """Where a file lives under MEDIA_DIR (and under /media in URLs)."""
    return f"{sha256[:2]}/{sha256}.{ext}"


def path(sha256: str, ext: str) -> Path:
    """The stored file on disk."""
    return config.MEDIA_DIR / _relpath(sha256, ext)


def url(sha256: str, ext: str) -> str:
    return f"/media/{_relpath(sha256, ext)}"


def thumb_url(sha256: str) -> str:
    return url(sha256, "thumb.webp")


def store(conn: sqlite3.Connection, data: bytes) -> int:
    """Store an image (if new) and return its media id. Caller commits.

    EXIF rotation is applied here, so stored pixels are upright and width/height
    are true; annotations then never depend on a rotation flag.
    Raises NotAnImage for anything that isn't a supported image.
    """
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise NotAnImage from None
    with image:
        ext = EXTENSIONS.get(image.format)
        if ext is None:
            raise NotAnImage(image.format)
        # Only a real rotation flag changes the pixels; otherwise the original bytes are kept as they are.
        upright = image
        if image.getexif().get(ExifTags.Base.Orientation, 1) != 1:
            upright = ImageOps.exif_transpose(image)
            buffer = io.BytesIO()
            upright.save(buffer, format=image.format, quality=95)
            data = buffer.getvalue()
        width, height = upright.size
        sha256 = hashlib.sha256(data).hexdigest()

        # Files first, then the row: a crash in between leaves only unused files.
        # Existing files are kept, so a missing thumbnail is also recreated here.
        file = path(sha256, ext)
        if not file.exists():
            _write(file, data)
        thumb = path(sha256, "thumb.webp")
        if not thumb.exists():
            _write(thumb, _thumbnail(upright))

    row = conn.execute("SELECT id FROM media WHERE sha256 = ?", (sha256,)).fetchone()
    if row:
        return row["id"]
    return conn.execute(
        "INSERT INTO media (sha256, ext, width, height) VALUES (?, ?, ?, ?)",
        (sha256, ext, width, height),
    ).lastrowid


def keep(conn: sqlite3.Connection, view_id: str, media_ids: set[int]) -> None:
    """Count these media as used by a report (a view), so remove_unused keeps them. Caller commits."""
    conn.execute("INSERT OR IGNORE INTO view_media (view_id, media_id) SELECT ?, id FROM media "
                 "WHERE id IN (SELECT value FROM json_each(?))", (view_id, json.dumps(sorted(media_ids))))


def release(conn: sqlite3.Connection, view_id: str) -> None:
    """A report no longer uses its media (remove_unused). Caller commits."""
    used = [r[0] for r in conn.execute("DELETE FROM view_media WHERE view_id = ? RETURNING media_id", (view_id,))]
    remove_unused(conn, used)


def remove_unused(conn: sqlite3.Connection, media_ids: list[int]) -> None:
    """Of these media, delete the rows no project, run or report uses any more. Caller commits.
    Their files stay while the history can put the rows back; history.purge removes them after."""
    if not media_ids:
        return
    conn.execute(
        """DELETE FROM media WHERE id IN (SELECT value FROM json_each(?))
           AND NOT EXISTS (SELECT 1 FROM project_media WHERE media_id = media.id)
           AND NOT EXISTS (SELECT 1 FROM run_result WHERE media_id = media.id)
           AND NOT EXISTS (SELECT 1 FROM view_media WHERE media_id = media.id)""",
        (json.dumps(media_ids),),
    )


def _thumbnail(image: Image.Image) -> bytes:
    # draft: a JPEG not yet loaded can be decoded at a fraction of its size, much faster for big photos.
    image.draft("RGB", (THUMB_SIZE, THUMB_SIZE))
    small = image.copy()
    small.thumbnail((THUMB_SIZE, THUMB_SIZE))
    if small.mode not in ("RGB", "RGBA"):  # e.g. palette, greyscale 16-bit, CMYK
        small = small.convert("RGBA" if "A" in small.getbands() else "RGB")
    buffer = io.BytesIO()
    small.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue()


def _write(path: Path, data: bytes) -> None:
    """Write via a temporary file, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
