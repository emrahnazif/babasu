-- Project and media ids are never reused (AUTOINCREMENT, as every other table has): a link, a
-- chat message or a log naming a deleted one never points at a newer one. SQLite can't add it to
-- a table, so both are rebuilt with the same columns and rows (db.migrate runs this with foreign
-- keys off and checks them before committing; the tables referring to these keep their rows).
-- annotation_source (009) reads media and project_media's triggers read it, so they're dropped
-- first and made again, unchanged, after.

DROP TRIGGER annotation_insert;
DROP TRIGGER annotation_update;
DROP VIEW annotation_source;

CREATE TABLE new_project (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  root_id     INTEGER REFERENCES project(id),  -- the original this is a version of; NULL for originals
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  classes     TEXT NOT NULL DEFAULT '[]',      -- JSON: [{"id", "name", "color", "active", "prompt"}]
  task        TEXT NOT NULL DEFAULT 'detection',  -- projects.TASKS
  annotator   TEXT,                            -- an id in annotators.FINDERS; NULL: none
  frozen_at   TEXT,                            -- set once; a frozen project never changes
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;
INSERT INTO new_project (id, root_id, name, description, classes, task, annotator, frozen_at, created_at, updated_at)
  SELECT id, root_id, name, description, classes, task, annotator, frozen_at, created_at, updated_at FROM project;
DROP TABLE project;
ALTER TABLE new_project RENAME TO project;

CREATE TABLE new_media (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256      TEXT NOT NULL UNIQUE,            -- of the stored file; also its name on disk
  ext         TEXT NOT NULL,
  width       INTEGER NOT NULL,
  height      INTEGER NOT NULL,
  uploaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;
INSERT INTO new_media (id, sha256, ext, width, height, uploaded_at)
  SELECT id, sha256, ext, width, height, uploaded_at FROM media;
DROP TABLE media;
ALTER TABLE new_media RENAME TO media;

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
