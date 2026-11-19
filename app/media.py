"""Media: an image in a project is a row of `media`; its file is stored once in MEDIA_DIR, named by
the SHA-256 of its content, whichever projects have it.

Each image also gets a small WebP thumbnail next to it (<sha256>.thumb.webp) for grids. A file
goes when no row and no history entry names it any more (history.purge).
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


def store(conn: sqlite3.Connection, project_id: int, data: bytes, name: str) -> tuple[int, bool]:
    """Store an image in a project under the file name it came with: its media id, and whether it's
    new there (a project has a file once: the same file again keeps the row it has). Caller commits.

    The file is written once for every project (named by its content), with its thumbnail.
    EXIF rotation is applied here, so stored pixels are upright and width/height are true;
    annotations then never depend on a rotation flag.
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
            _write(thumb, _thumbnail(image))

    row = conn.execute("SELECT id FROM media WHERE project_id = ? AND sha256 = ?", (project_id, sha256)).fetchone()
    if row:
        return row["id"], False
    return conn.execute(
        "INSERT INTO media (project_id, sha256, ext, width, height, name) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, sha256, ext, width, height, name),
    ).lastrowid, True


def listed(conn: sqlite3.Connection, media_ids: list[int]) -> list[dict]:
    """These images, in the order given (those that exist): media_id, project_id, name, width,
    height and thumbnail."""
    rows = {r["id"]: r for r in conn.execute(
        "SELECT id, project_id, sha256, width, height, name FROM media WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(media_ids),))}
    return [{"media_id": m, "project_id": rows[m]["project_id"], "name": rows[m]["name"], "width": rows[m]["width"],
             "height": rows[m]["height"], "thumb_url": thumb_url(rows[m]["sha256"])} for m in media_ids if m in rows]


def thumbnail(file: Path) -> bytes:
    """A thumbnail of an image file that isn't stored (one about to be imported or attached), made
    as it's asked for. Raises NotAnImage for a file Pillow can't read."""
    try:
        with Image.open(file) as image:
            return _thumbnail(image)
    except OSError:  # UnidentifiedImageError, or a broken file
        raise NotAnImage from None


def _thumbnail(image: Image.Image) -> bytes:
    """An image as opened, as a small upright WebP. It shrinks the image itself: no full-size copy."""
    # draft: a JPEG not yet loaded can be decoded at a fraction of its size, much faster for big photos.
    image.draft("RGB", (THUMB_SIZE, THUMB_SIZE))
    image.thumbnail((THUMB_SIZE, THUMB_SIZE))
    small = ImageOps.exif_transpose(image)  # upright if its EXIF says it's turned
    if small.mode not in ("RGB", "RGBA"):  # e.g. palette, greyscale 16-bit, CMYK
        small = small.convert("RGBA" if "A" in small.getbands() else "RGB")
    buffer = io.BytesIO()
    small.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue()


PICTURE_SIZE = 768  # pixels a picture for a model is at most on its longer side: enough to judge, few tokens


def picture(image: Image.Image, box: tuple[float, float, float, float] | None = None) -> bytes:
    """An image (opened, or drawn on), or a part of it (box: x1, y1, x2, y2 in its pixels), scaled
    to fit PICTURE_SIZE, as a JPEG: what a model is shown."""
    if box is None:
        image.draft("RGB", (PICTURE_SIZE, PICTURE_SIZE))  # decoded small, as for thumbnails
    else:
        image = image.crop(box)
    image.thumbnail((PICTURE_SIZE, PICTURE_SIZE))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _write(path: Path, data: bytes) -> None:
    """Write via a temporary file, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
