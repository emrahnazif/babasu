"""search_docs by meaning: each part of the docs and each API operation it searches, as a vector from
mozo's text embedding model, so a query finds the parts that mean what it asks even in other words
("annotate run" finds POST /api/annotate). search_docs ranks by words too and combines the two.

Built on the person's machine, not shipped: the model downloads at the app's start (a job, as any
model's), then a job embeds every part and keeps the vectors in INDEX, each with a hash of its text
and the model's revision (vectors of other weights don't compare). The next start embeds only what
changed (an update's docs, API or pixelflow), or nothing. Until it's built, search_docs ranks by
words alone. A query is embedded by the model loaded as any vision model is (vision.load): loaded at
the first search, and listed with the others while it's in memory.
"""

import hashlib
import io
import sqlite3

import numpy as np
from mozo import weights

from app import config, jobs, media, vision

FAMILY, VARIANT = "granite_embedding_r2", "small"
MODEL = {"family": FAMILY, "variant": VARIANT}
INDEX = config.DATA_DIR / "search" / "docs.npz"
KIND = "docs_index"
BATCH = 16  # parts embedded at a time

_matrix: np.ndarray | None = None  # each part's vector, in the order of _texts(), once they're all embedded


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _texts() -> list[str]:
    from app import agent_tools  # here, not at the top: it imports this module
    return agent_tools.meaning_texts()


def _stored() -> dict[str, np.ndarray]:
    """INDEX's vectors by their text's hash: none if there's no index, or it was built with other
    weights than those the model loads now (its vectors wouldn't compare)."""
    if not INDEX.exists():
        return {}
    with np.load(INDEX) as found:
        if str(found["revision"]) != weights.revision_of(FAMILY, VARIANT):
            return {}
        return dict(zip(found["hashes"].tolist(), found["vectors"]))


def _missing(stored: dict[str, np.ndarray]) -> list[str]:
    """The parts that have no vector in `stored`."""
    return [text for text in _texts() if _hash(text) not in stored]


def start(conn: sqlite3.Connection) -> None:
    """At the app's start: build the index, or bring it up to date, unless every part is embedded;
    the job waits for the model's download."""
    if _missing(_stored()) and not jobs.find_active(conn, KIND, MODEL):
        jobs.start(conn, KIND, MODEL)


@jobs.kind(KIND, lane="gpu", title="Index the docs for search", needs=vision.download_first)
def build(ctx: jobs.Context, family: str, variant: str) -> dict:
    global _matrix
    vectors = _stored()
    new = list(dict.fromkeys(_missing(vectors)))
    ctx.total(len(new))
    model = vision.load(family, variant)
    for at in range(0, len(new), BATCH):
        if ctx.cancelled:
            return {"embedded": at}
        batch = new[at:at + BATCH]
        vectors |= dict(zip(map(_hash, batch), model.encode_text(batch)))
        ctx.advance(len(batch))
    texts = _texts()
    kept = {_hash(text): vectors[_hash(text)] for text in texts}  # what the docs no longer have goes
    written = io.BytesIO()
    np.savez(written, revision=weights.revision_of(family, variant), hashes=np.array(list(kept)),
             vectors=np.array(list(kept.values())))
    media.write(INDEX, written.getvalue())
    _matrix = np.stack([kept[_hash(text)] for text in texts])
    return {"embedded": len(new), "parts": len(kept)}


def ranking(query: str) -> list[int] | None:
    """The parts' positions (in search_docs' order), the closest in meaning to `query` first; None
    while they aren't all embedded, or the model was removed."""
    global _matrix
    if _matrix is None and not _missing(stored := _stored()):
        _matrix = np.stack([stored[_hash(text)] for text in _texts()])
    if _matrix is None or not vision.downloaded(FAMILY, VARIANT):
        return None
    return np.argsort(-(_matrix @ vision.load(FAMILY, VARIANT).encode_text(query)[0])).tolist()
