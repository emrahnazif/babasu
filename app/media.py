"""Media files: stored once in MEDIA_DIR, named by the SHA-256 of their content."""

import hashlib
import io
import sqlite3

from PIL import Image, ImageOps

from app import config

# Pillow format name -> file extension.
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif"}


def _relpath(sha256: str, ext: str) -> str:
    """Where a file lives under MEDIA_DIR (and under /media in URLs)."""
    return f"{sha256[:2]}/{sha256}.{ext}"


def url(sha256: str, ext: str) -> str:
    return f"/media/{_relpath(sha256, ext)}"


def store(conn: sqlite3.Connection, data: bytes) -> int:
    """Store an image (if new) and return its media id. Caller commits.

    EXIF rotation is applied here, so stored pixels are upright and width/height
    are true; annotations then never depend on a rotation flag.
    """
    with Image.open(io.BytesIO(data)) as image:
        ext = EXTENSIONS[image.format]
        upright = ImageOps.exif_transpose(image)
        if upright is not image:  # it was rotated: store the upright pixels instead
            buffer = io.BytesIO()
            upright.save(buffer, format=image.format)
            data = buffer.getvalue()
        width, height = upright.size

    sha256 = hashlib.sha256(data).hexdigest()
    row = conn.execute("SELECT id FROM media WHERE sha256 = ?", (sha256,)).fetchone()
    if row:
        return row["id"]

    # File first, then the row: a crash in between leaves only an unused file.
    path = config.MEDIA_DIR / _relpath(sha256, ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)

    return conn.execute(
        "INSERT INTO media (sha256, ext, width, height) VALUES (?, ?, ?, ?)",
        (sha256, ext, width, height),
    ).lastrowid
