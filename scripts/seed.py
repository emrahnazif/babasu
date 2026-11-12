"""Placeholder data for development: a project of generated shape images with matching annotations.

Squares have a box, circles a box and a center keypoint, triangles a box and a polygon.

    uv run python -m scripts.seed

Writes to the real workspace (DATA_DIR). Runs once: does nothing if the project exists.
The images are the same every run (fixed random seed), so media is never stored twice.
"""

import io
import random
from contextlib import closing

from PIL import Image, ImageDraw

from app import db, media, projects

NAME = "Shapes (sample)"
CLASSES = [
    {"id": 0, "name": "square", "prompt": "square", "color": "#e11d48", "active": True},
    {"id": 1, "name": "circle", "prompt": "circle", "color": "#2563eb", "active": True},
    {"id": 2, "name": "triangle", "prompt": "triangle", "color": "#16a34a", "active": True},
]
IMAGE_COUNT = 24
WIDTH, HEIGHT = 640, 480


def make_image(rng: random.Random) -> tuple[bytes, list[dict]]:
    """Draw 1-4 shapes on a plain background; return JPEG bytes and their annotations."""
    image = Image.new("RGB", (WIDTH, HEIGHT), tuple(rng.randint(200, 245) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    annotations = []
    for _ in range(rng.randint(1, 4)):
        class_id = rng.randrange(len(CLASSES))
        size = rng.randint(60, 160)
        x, y = rng.randint(0, WIDTH - size), rng.randint(0, HEIGHT - size)
        fill = tuple(rng.randint(40, 160) for _ in range(3))
        # One object per shape, with every form it has (like COCO and pixelflow's Detection).
        obj = {"class_id": class_id, "bbox": [x, y, size, size]}
        box = (x, y, x + size, y + size)
        if class_id == 0:
            draw.rectangle(box, fill=fill)
        elif class_id == 1:
            draw.ellipse(box, fill=fill)
            obj["keypoints"] = [{"name": "center", "point": [x + size / 2, y + size / 2]}]
        else:
            obj["segments"] = [[x + size / 2, y], [x + size, y + size], [x, y + size]]
            draw.polygon([tuple(p) for p in obj["segments"]], fill=fill)
        annotations.append(obj)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), annotations


def main() -> None:
    db.migrate()
    with closing(db.connect()) as conn, conn:  # `conn` as a context manager commits at the end
        if conn.execute("SELECT 1 FROM project WHERE name = ?", (NAME,)).fetchone():
            print(f"'{NAME}' already exists; nothing to do.")
            return
        project_id = projects.create(conn, NAME, "Generated squares, circles and triangles.", "detection", CLASSES)
        rng = random.Random(0)
        for i in range(IMAGE_COUNT):
            data, annotations = make_image(rng)
            split = "train" if i < 18 else "val" if i < 22 else "test"
            projects.add_image(conn, project_id, media.store(conn, data), f"shapes_{i:03}.jpg", annotations, split)
    print(f"Created '{NAME}' with {IMAGE_COUNT} images.")


if __name__ == "__main__":
    main()
