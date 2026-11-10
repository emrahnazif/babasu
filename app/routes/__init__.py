"""Route modules (one per area), plus what they share."""

from pydantic import ValidationError


def safe_next(next: str | None) -> str:
    """A return target from a form or query: only local paths are allowed, anything else is "/"."""
    if next and next.startswith("/") and not next.startswith("//"):
        return next
    return "/"


def message(exc: ValueError) -> str:
    """A failed change explained in one line: a pydantic error's first message, or the error's text."""
    if isinstance(exc, ValidationError):
        error = exc.errors(include_url=False)[0]
        field = ".".join(str(part) for part in error["loc"])
        text = error["msg"].removeprefix("Value error, ")
        return f"{field}: {text}" if field else text
    return str(exc)
