"""Importing images from files and folders (local disk or NAS), as a background job: into a project,
or attached to the chat message they were dropped with.

Crash-safe and safe to retry (product.md "Jobs"): each image is stored and added in its own
transaction, and an image the project already has is skipped, so a retry only does what's left.
"""

import os
from contextlib import closing
from pathlib import Path

from app import db, jobs, media, projects

KIND = "import_images"


def find_images(paths: list[str]) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[str]]:
    """The images the paths name, as (file, name in the project); the files given that aren't
    images by their name (e.g. notes.txt, or a photo saved without a suffix), the same way; and
    the paths that don't exist.

    A folder contributes its image files and those of its subfolders, named relative to it
    (e.g. "train/0001.jpg"); hidden files and folders are left out (e.g. .DS_Store and the
    "._name.jpg" files macOS leaves on network drives).
    os.walk gets file types from the directory listing itself and prunes hidden folders before
    entering them: on a NAS each extra look at a file is a network round trip.
    """
    found, others, missing = [], [], []
    for given in paths:
        path = Path(given)
        if path.is_file():
            (found if path.suffix.lower() in media.SUFFIXES else others).append((path, path.name))
        elif path.is_dir():
            in_folder = []
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                in_folder += [
                    Path(root, f) for f in files if not f.startswith(".") and Path(f).suffix.lower() in media.SUFFIXES
                ]
            found += [(file, file.relative_to(path).as_posix()) for file in sorted(in_folder)]
        else:
            missing.append(given)
    return found, others, missing


@jobs.kind(KIND, lane="io", title="Import images")
def import_images(ctx: jobs.Context, paths: list[str], project_id: int | None = None, message_id: int | None = None) -> dict:
    ctx.advance(0, "Looking for images…")  # the total isn't known yet: the bar is indeterminate
    images, others, missing = find_images(paths)
    files = images + others  # a file given directly is tried all the same: its content decides
    for path in missing:
        ctx.log(f"Not found: {path}")
    if not files and missing:
        raise ValueError(f"Not found: {', '.join(missing)}")
    ctx.total(len(files))

    added = duplicates = 0
    skipped = len(missing)
    with closing(db.connect()) as conn:
        for file, name in files:
            if ctx.cancelled:
                break
            try:
                data = file.read_bytes()
                with conn:  # one image, one transaction
                    media_id = media.store(conn, data, name)
                    is_new = (media.attach(conn, message_id, media_id) if project_id is None
                              else projects.add_image(conn, project_id, media_id, name))
            except (media.NotAnImage, OSError) as exc:
                skipped += 1
                ctx.log(f"Skipped {name}: {str(exc) or 'not a readable image'}")
            else:
                if is_new:
                    added += 1
                else:
                    duplicates += 1
            ctx.advance(1, name)

    return {"added": added, "already_there": duplicates, "skipped": skipped}
