"""Route modules (one per area), plus what they share."""


def safe_next(next: str | None) -> str:
    """A return target from a form or query: only local paths are allowed, anything else is "/"."""
    if next and next.startswith("/") and not next.startswith("//"):
        return next
    return "/"
