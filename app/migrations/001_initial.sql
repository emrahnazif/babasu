-- Projects, media files, and which media each project uses (with its annotations).
-- See product.md "Data" for the rules behind this schema.

CREATE TABLE project (
  id          INTEGER PRIMARY KEY,
  root_id     INTEGER REFERENCES project(id),  -- the original this is a version of; NULL for originals
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  classes     TEXT NOT NULL DEFAULT '[]',      -- JSON: [{"id", "name", "color", "active"}]
  frozen_at   TEXT,                            -- set once; a frozen project never changes
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE TABLE media (
  id          INTEGER PRIMARY KEY,
  sha256      TEXT NOT NULL UNIQUE,            -- of the stored file; also its name on disk
  ext         TEXT NOT NULL,
  width       INTEGER NOT NULL,
  height      INTEGER NOT NULL,
  uploaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE TABLE project_media (
  project_id  INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  media_id    INTEGER NOT NULL REFERENCES media(id),  -- a media row can't be deleted while used
  filename    TEXT NOT NULL,
  split       TEXT,                            -- 'train', 'val', 'test' or NULL
  annotations TEXT NOT NULL DEFAULT '[]',      -- JSON: [{"class_id", "type", "bbox" | "segments"}]
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  PRIMARY KEY (project_id, media_id)
) STRICT, WITHOUT ROWID;

CREATE INDEX project_media_media_id ON project_media(media_id);
