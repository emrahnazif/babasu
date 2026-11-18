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


def store(conn: sqlite3.Connection, data: bytes, name: str) -> int:
    """Store an image (if new) under the file name it came with, and return its media id. Caller
    commits.

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
            _write(thumb, _thumbnail(image))

    row = conn.execute("SELECT id FROM media WHERE sha256 = ?", (sha256,)).fetchone()
    if row:
        return row["id"]
    return conn.execute(
        "INSERT INTO media (sha256, ext, width, height, name) VALUES (?, ?, ?, ?, ?)",
        (sha256, ext, width, height, name),
    ).lastrowid


def remove_unused(conn: sqlite3.Connection, media_ids: list[int]) -> None:
    """Of these media, delete the rows no project, run or chat message (attached) uses any more.
    Caller commits. Their files stay while the history can put the rows back; history.purge
    removes them after."""
    if not media_ids:
        return
    conn.execute(
        """DELETE FROM media WHERE id IN (SELECT value FROM json_each(?))
           AND NOT EXISTS (SELECT 1 FROM project_media WHERE media_id = media.id)
           AND NOT EXISTS (SELECT 1 FROM run_result WHERE media_id = media.id)
           AND NOT EXISTS (SELECT 1 FROM attachment WHERE media_id = media.id)""",
        (json.dumps(media_ids),),
    )


def attach(conn: sqlite3.Connection, message_id: int, media_id: int) -> bool:
    """Attach a stored image to a chat message; False if it already was. Caller commits."""
    return conn.execute("INSERT OR IGNORE INTO attachment (message_id, media_id) VALUES (?, ?)", (message_id, media_id)).rowcount == 1


def attached(conn: sqlite3.Connection, message_id: int, limit: int = -1) -> list[int]:
    """The images attached to a chat message, in the order they came (the first `limit`)."""
    return [row[0] for row in conn.execute("SELECT media_id FROM attachment WHERE message_id = ? ORDER BY rowid LIMIT ?",
                                           (message_id, limit))]


# An image's name, in SQL over `media m LEFT JOIN project_media pm` (the project it's seen in, if
# any): its name there; else its own, the file name it came with; else the start of its sha256.
NAME = "coalesce(pm.filename, m.name, substr(m.sha256, 1, 12))"


def listed(conn: sqlite3.Connection, media_ids: list[int], project_id: int | None = None) -> list[dict]:
    """These stored images, in the order given (those that are stored): media_id, name (NAME, in
    the project given), width, height and thumbnail."""
    rows = {r["id"]: r for r in conn.execute(
        f"""SELECT m.id, m.sha256, m.width, m.height, {NAME} AS name
            FROM media m LEFT JOIN project_media pm ON pm.media_id = m.id AND pm.project_id = ?
            WHERE m.id IN (SELECT value FROM json_each(?))""", (project_id, json.dumps(media_ids)))}
    return [{"media_id": m, "name": rows[m]["name"], "width": rows[m]["width"], "height": rows[m]["height"],
             "thumb_url": thumb_url(rows[m]["sha256"])} for m in media_ids if m in rows]


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
