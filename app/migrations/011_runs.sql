-- Workflow runs and what they kept (app/runs.py). A run is one workflow over a set of stored
-- images; its results are the detections that reached its save_to_babasu nodes, one row per
-- image and label, written as each image finishes.

CREATE TABLE run (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id      INTEGER REFERENCES job(id) ON DELETE SET NULL,      -- the job that ran it
  workflow_id TEXT NOT NULL,                                       -- runs.workflows() id, when it ran
  workflow    TEXT NOT NULL,                                       -- JSON: the workflow as it ran (mozo's {nodes, edges})
  project_id  INTEGER REFERENCES project(id) ON DELETE SET NULL,  -- the project its images came from, if any
  name        TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE TABLE run_result (
  run_id      INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  media_id    INTEGER NOT NULL REFERENCES media(id),  -- a media row can't be deleted while a run uses it
  label       TEXT NOT NULL,                          -- the save_to_babasu node's label (or its id)
  result      TEXT NOT NULL,                          -- JSON: one mozo annotation_line (detections, width, height)
  PRIMARY KEY (run_id, media_id, label)
) STRICT, WITHOUT ROWID;

CREATE INDEX run_result_media_id ON run_result(media_id);
