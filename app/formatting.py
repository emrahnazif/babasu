"""Values for people: numbers, sizes, times and icons, as the app's pages show them (components.html
uses them). It imports nothing of the app but its paths."""

import re
from datetime import UTC, datetime
from functools import cache

from markupsafe import Markup

from app import config

ICONS_DIR = config.STATIC_DIR / "icons"


@cache
def icon(name: str) -> Markup:
    """Inline an SVG from static/icons, so it takes the surrounding text colour."""
    path = ICONS_DIR / f"{name}.svg"
    if not re.fullmatch(r"[a-z0-9-]+", name) or not path.is_file():
        raise ValueError(f"there's no icon named {name!r} (static/icons)")
    return Markup(path.read_text())


def number(n: int | float) -> str:
    """A number for people: thousands grouped, a fraction to three significant digits: "5,084",
    "0.912", "3.4", "-6.5"."""
    if isinstance(n, float) and not n.is_integer() and abs(n) < 1000:
        return f"{n:,.3g}"
    return f"{n:,.0f}"


def plural(n: int, word: str) -> str:
    """A count with its noun: "1 image", "1,240 images", "3 classes"."""
    return f"{number(n)} {word if n == 1 else word + ('es' if word.endswith('s') else 's')}"


def filesize(n: int) -> str:
    """Bytes for people, in decimal units as downloads are shown: "820 bytes", "56.1 MB", "3.45 GB"."""
    if n < 1000:
        return f"{n:,} bytes"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1000
        if n < 1000 or unit == "TB":
            break
    return f"{n:.{0 if n >= 100 else 1 if n >= 10 else 2}f} {unit}"


def counted(job: dict) -> str:
    """How far a job is, for people: "1.20 GB / 3.45 GB" when it counts bytes, else "120 / 5,084"."""
    if job["unit"] == "bytes":
        return f"{filesize(job['done'])} / {filesize(job['total'])}"
    return f"{number(job['done'])} / {number(job['total'])}"


def ago(timestamp: str) -> str:
    """A short "how long ago" for a stored UTC time: now, 5m, 3h, Yesterday, 4d, then the date."""
    then = datetime.fromisoformat(timestamp)
    seconds = (datetime.now(UTC) - then).total_seconds()
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    if seconds < 2 * 86400:
        return "Yesterday"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d"
    local = then.astimezone()
    return f"{local:%b} {local.day}"


def period(timestamp: str) -> str:
    """Which stretch of time a stored UTC time falls in, for grouping lists: Today, Yesterday,
    Previous 7 days, Previous 30 days, Older (by local calendar days)."""
    days = (datetime.now().astimezone().date() - datetime.fromisoformat(timestamp).astimezone().date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "Previous 7 days"
    if days < 30:
        return "Previous 30 days"
    return "Older"


# What templates use them by: filters ({{ n | number }}, {{ count | plural("image") }}) and globals
# ({{ icon("check") }}). components.html needs them.
def clock(seconds: float, tenths: bool = False) -> str:
    """A length or a moment of a video, as a player shows it: "0:42", "12:05", "1:02:09"; with
    `tenths`, "0:42.5" (js/video.js writes the same)."""
    whole = int(seconds)
    hours, minutes = whole // 3600, whole % 3600 // 60
    rest = f"{seconds % 60:04.1f}" if tenths else f"{whole % 60:02}"
    return f"{hours}:{minutes:02}:{rest}" if hours else f"{minutes}:{rest}"


FILTERS = {"number": number, "plural": plural, "filesize": filesize, "counted": counted, "ago": ago, "period": period,
           "clock": clock}
GLOBALS = {"icon": icon}
