"""Media: an image or a video in a project is a row of `media`; its file is stored once in
MEDIA_DIR, named by the SHA-256 of its content, whichever projects have it. A video's file is the
original, as it came: never re-encoded.

Each file also gets a small WebP thumbnail next to it (<sha256>.thumb.webp; a video's is a frame),
and a video its frames' times (<sha256>.frames.json: what reading a frame exactly needs,
pf.VideoReader.read_frame). A file goes when no row and no history entry names it any more
(history.purge).
"""

import hashlib
import io
import json
import secrets
import sqlite3
from pathlib import Path

import pixelflow as pf
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from app import config

# The supported image formats: Pillow format name -> extension of the stored file.
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif"}
# What a file is, by its file name: suffix -> MIME type. The videos are what OpenCV (so pixelflow
# and mozo) reads.
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
               ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/x-m4v", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
               ".mkv": "video/x-matroska", ".webm": "video/webm", ".mpg": "video/mpeg", ".mpeg": "video/mpeg",
               ".wmv": "video/x-ms-wmv", ".flv": "video/x-flv", ".ts": "video/mp2t"}
SUFFIXES = set(IMAGE_TYPES) | set(VIDEO_TYPES)  # file names to look for
FORMATS_TEXT = "JPEG, PNG, WebP, BMP and TIFF images; MP4, MOV, MKV, AVI and WebM videos"  # for people
THUMB_SIZE = 256  # longest side, in pixels
SAMPLE_FPS = 2.0  # the rate a new video is worked at (its own if it's slower)


class NotMedia(ValueError):
    """The file isn't an image or a video in one of the supported formats."""


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_TYPES


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
    Raises NotMedia for anything that isn't a supported image.
    """
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise NotMedia("not an image") from None
    with image:
        ext = EXTENSIONS.get(image.format)
        if ext is None:
            raise NotMedia(f"not a supported image ({image.format})")
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

    return _row(conn, project_id, sha256, ext, IMAGE_TYPES[f".{ext}"], width, height, name)


def _row(conn: sqlite3.Connection, project_id: int, sha256: str, ext: str, type: str, width: int, height: int,
         name: str, video: dict | None = None) -> tuple[int, bool]:
    """The project's row for a stored file: the one it has, or a new one (`video`: duration,
    frame_count, fps, sample_fps). Its id, and whether it's new."""
    row = conn.execute("SELECT id FROM media WHERE project_id = ? AND sha256 = ?", (project_id, sha256)).fetchone()
    if row:
        return row["id"], False
    video = video or {}
    return conn.execute(
        """INSERT INTO media (project_id, sha256, ext, type, width, height, name, duration, frame_count, fps, sample_fps)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project_id, sha256, ext, type, width, height, name, video.get("duration"), video.get("frame_count"),
         video.get("fps"), video.get("sample_fps")),
    ).lastrowid, True


def store_file(conn: sqlite3.Connection, project_id: int, file: Path, name: str) -> tuple[int, bool]:
    """Store an image or a video file in a project under `name`, as store() and store_video() do.
    Raises NotMedia for anything that isn't one, OSError for a file that can't be read."""
    if is_video(file.name):
        return store_video(conn, project_id, file, name)
    return store(conn, project_id, file.read_bytes(), name)


def store_video(conn: sqlite3.Connection, project_id: int, file: Path, name: str) -> tuple[int, bool]:
    """Store a video in a project under the file name it came with: its media id, and whether it's
    new there (as store()). The original file is copied as it is, hashed as it's copied, a block at
    a time (a video can be gigabytes, on a NAS), never re-encoded; its frames' times are kept beside
    it (frames()) and its thumbnail is its first frame. A new row works at SAMPLE_FPS, or the
    video's own rate if slower. Caller commits. Raises NotMedia for a file OpenCV can't read as a video."""
    ext = file.suffix.lower().lstrip(".")
    config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.MEDIA_DIR / f".{secrets.token_hex(8)}.tmp"
    digest = hashlib.sha256()
    with file.open("rb") as source, tmp.open("wb") as copy:
        while block := source.read(1 << 20):
            digest.update(block)
            copy.write(block)
    sha256 = digest.hexdigest()
    stored = path(sha256, ext)
    if stored.exists():  # another project has it already
        tmp.unlink()
    else:
        stored.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(stored)
    first, fps = _first_frame(stored)
    times = frames(sha256, ext)
    thumb = path(sha256, "thumb.webp")
    if not thumb.exists():
        _write(thumb, _thumbnail(pf.to_pil(first)))
    fps = fps or len(times) / (times[-1] / 1000 or 1)  # a file without a rate: its frames over its length
    height, width = first.shape[:2]  # upright: OpenCV applies a rotation flag
    return _row(conn, project_id, sha256, ext, VIDEO_TYPES[file.suffix.lower()], width, height, name,
                {"duration": times[-1] / 1000 + 1 / fps, "frame_count": len(times), "fps": fps,
                 "sample_fps": min(SAMPLE_FPS, fps)})


def _first_frame(file: Path) -> tuple:
    """A video file's first frame (RGB) and its frame rate (None if it doesn't say). Raises NotMedia
    for a file OpenCV can't read as a video, or one with no frames."""
    try:
        with pf.VideoReader(str(file)) as video:
            first = video.read()
    except ValueError as e:  # OpenCV can't open it
        raise NotMedia(str(e)) from None
    if first is None:
        raise NotMedia("a video with no frames")
    return first, video.fps


def frames(sha256: str, ext: str) -> list[float]:
    """A stored video's frames' times, in milliseconds: kept beside the file, worked out once (a
    pass over it) the first time."""
    kept = path(sha256, "frames.json")
    if kept.exists():
        return json.loads(kept.read_text())
    with pf.VideoReader(str(path(sha256, ext))) as video:
        times = video.timestamps()
    _write(kept, json.dumps(times).encode())
    return times


def listed(conn: sqlite3.Connection, media_ids: list[int]) -> list[dict]:
    """These images, in the order given (those that exist): media_id, project_id, name, width,
    height and thumbnail."""
    rows = {r["id"]: r for r in conn.execute(
        "SELECT id, project_id, sha256, width, height, name FROM media WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(media_ids),))}
    return [{"media_id": m, "project_id": rows[m]["project_id"], "name": rows[m]["name"], "width": rows[m]["width"],
             "height": rows[m]["height"], "thumb_url": thumb_url(rows[m]["sha256"])} for m in media_ids if m in rows]


def thumbnail(file: Path) -> bytes:
    """A thumbnail of an image or video file that isn't stored (one about to be imported), made as
    it's asked for: a video's first frame. Raises NotMedia for a file that can't be read as one."""
    if is_video(file.name):
        return _thumbnail(pf.to_pil(_first_frame(file)[0]))
    try:
        with Image.open(file) as image:
            return _thumbnail(image)
    except OSError as e:  # not readable as an image (UnidentifiedImageError), or a broken file
        raise NotMedia(str(e)) from None


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
