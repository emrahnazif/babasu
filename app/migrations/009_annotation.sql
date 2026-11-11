-- One row per annotation, with its geometry in pixels, so filters and statistics can ask about
-- objects ("a car between 8 and 16 px", "touching the edge") with indexes instead of reading
-- every image's JSON. project_media.annotations stays the source of truth: triggers rebuild an
-- image's rows whenever its annotations are written, so no write path can forget them.

-- Each annotation of each image, measured: its box (an annotation without a bbox gets the box
-- around its polygon or keypoints), and whether it touches the image's edge (within 1 px).
CREATE VIEW annotation_source AS
SELECT project_id, media_id, idx, class_id,
       x0 AS x, y0 AS y, x1 - x0 AS w, y1 - y0 AS h,
       (x0 <= 1 OR y0 <= 1 OR x1 >= image_width - 1 OR y1 >= image_height - 1) AS edge
FROM (
  SELECT pm.project_id, pm.media_id, a.key AS idx, a.value ->> 'class_id' AS class_id,
         m.width AS image_width, m.height AS image_height,
         coalesce(a.value -> 'bbox' ->> 0,
                  (SELECT min(p.value ->> 0) FROM json_each(a.value, '$.segments') p),
                  (SELECT min(k.value -> 'point' ->> 0) FROM json_each(a.value, '$.keypoints') k)) AS x0,
         coalesce(a.value -> 'bbox' ->> 1,
                  (SELECT min(p.value ->> 1) FROM json_each(a.value, '$.segments') p),
                  (SELECT min(k.value -> 'point' ->> 1) FROM json_each(a.value, '$.keypoints') k)) AS y0,
         coalesce((a.value -> 'bbox' ->> 0) + (a.value -> 'bbox' ->> 2),
                  (SELECT max(p.value ->> 0) FROM json_each(a.value, '$.segments') p),
                  (SELECT max(k.value -> 'point' ->> 0) FROM json_each(a.value, '$.keypoints') k)) AS x1,
         coalesce((a.value -> 'bbox' ->> 1) + (a.value -> 'bbox' ->> 3),
                  (SELECT max(p.value ->> 1) FROM json_each(a.value, '$.segments') p),
                  (SELECT max(k.value -> 'point' ->> 1) FROM json_each(a.value, '$.keypoints') k)) AS y1
  FROM project_media pm
  JOIN media m ON m.id = pm.media_id,
  json_each(pm.annotations) a
);

CREATE TABLE annotation (
  project_id  INTEGER NOT NULL,
  media_id    INTEGER NOT NULL,
  idx         INTEGER NOT NULL,                -- its position in the image's annotations list
  class_id    INTEGER NOT NULL,
  x           REAL NOT NULL,                   -- box in pixels of the original image
  y           REAL NOT NULL,
  w           REAL NOT NULL,
  h           REAL NOT NULL,
  edge        INTEGER NOT NULL,                -- 1 if it touches the image's edge
  area        REAL GENERATED ALWAYS AS (w * h) VIRTUAL,
  aspect      REAL GENERATED ALWAYS AS (w / nullif(h, 0)) VIRTUAL,  -- width / height
  PRIMARY KEY (project_id, media_id, idx),
  FOREIGN KEY (project_id, media_id) REFERENCES project_media (project_id, media_id) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE INDEX annotation_class ON annotation (project_id, class_id);

CREATE TRIGGER annotation_insert AFTER INSERT ON project_media BEGIN
  INSERT INTO annotation (project_id, media_id, idx, class_id, x, y, w, h, edge)
  SELECT project_id, media_id, idx, class_id, x, y, w, h, edge FROM annotation_source
  WHERE project_id = NEW.project_id AND media_id = NEW.media_id;
END;

CREATE TRIGGER annotation_update AFTER UPDATE OF annotations ON project_media BEGIN
  DELETE FROM annotation WHERE project_id = OLD.project_id AND media_id = OLD.media_id;
  INSERT INTO annotation (project_id, media_id, idx, class_id, x, y, w, h, edge)
  SELECT project_id, media_id, idx, class_id, x, y, w, h, edge FROM annotation_source
  WHERE project_id = NEW.project_id AND media_id = NEW.media_id;
END;

-- The images already there.
INSERT INTO annotation (project_id, media_id, idx, class_id, x, y, w, h, edge)
SELECT project_id, media_id, idx, class_id, x, y, w, h, edge FROM annotation_source;
