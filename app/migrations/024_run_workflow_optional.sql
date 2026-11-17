-- A run of a workflow not saved (a draft tried on a few images, its document given) has no workflow
-- id. SQLite can't change a column, so the table is rebuilt with the same columns and rows (db.migrate
-- runs this with foreign keys off; run_result's rows keep pointing at their runs).

CREATE TABLE new_run (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id      INTEGER REFERENCES job(id) ON DELETE SET NULL,      -- the job that ran it
  workflow_id TEXT,                                                -- runs.workflows() id, when it ran; empty: not saved
  workflow    TEXT NOT NULL,                                       -- JSON: the workflow as it ran (mozo's {nodes, edges})
  project_id  INTEGER REFERENCES project(id) ON DELETE SET NULL,  -- the project its images came from, if any
  name        TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  classes     TEXT,
  annotates   INTEGER NOT NULL DEFAULT 0 CHECK (annotates IN (0, 1))
) STRICT;

INSERT INTO new_run (id, job_id, workflow_id, workflow, project_id, name, created_at, classes, annotates)
  SELECT id, job_id, workflow_id, workflow, project_id, name, created_at, classes, annotates FROM run;
DROP TABLE run;
ALTER TABLE new_run RENAME TO run;
