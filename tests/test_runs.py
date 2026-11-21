"""What a run keeps of what its workflow made."""

from types import SimpleNamespace

import numpy as np

from app import db, media, projects, runs


def test_many_images_reaching_a_save_node_are_each_kept(conn, image):
    """A node before it made one image per object (crop_around_detections): each crop is a new image."""
    with db.action("request", "make"), conn:
        project_id = projects.create(conn, "Crops")
        media_id, _ = media.store(conn, project_id, image(0), "street.png")
    row = conn.execute("SELECT id, project_id, name FROM media WHERE id = ?", (media_id,)).fetchone()
    save = SimpleNamespace(id="save", spec=SimpleNamespace(name="save_image_to_babasu"), parameters={"label": "crop"})
    crops = [np.full((8, 8, 3), n * 40, np.uint8) for n in range(3)]
    with db.action("request", "run"), conn:
        runs._Keeper(conn, [save], [], 0).found(row, {"save": crops})
    names = [r[0] for r in conn.execute("SELECT name FROM media WHERE project_id = ? ORDER BY name", (project_id,))]
    assert names == ["street-crop-1.png", "street-crop-2.png", "street-crop-3.png", "street.png"]
