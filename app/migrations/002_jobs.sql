-- Background jobs (imports, training, ...). See app/jobs.py and product.md "Jobs".

CREATE TABLE job (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused: logs are named by id
  kind             TEXT NOT NULL,
  project_id       INTEGER REFERENCES project(id) ON DELETE SET NULL,
  params           TEXT NOT NULL DEFAULT '{}',          -- JSON: the job function's arguments
  status           TEXT NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted')),
  done             INTEGER NOT NULL DEFAULT 0,
  total            INTEGER,                             -- NULL until the job knows it
  message          TEXT NOT NULL DEFAULT '',            -- what it's doing now, e.g. the current file
  result           TEXT,                                -- JSON summary, when it finishes
  error            TEXT,
  stop_reason      TEXT CHECK (stop_reason IN ('user', 'shutdown')),  -- set to ask a running job to stop
  created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  started_at       TEXT,
  finished_at      TEXT
) STRICT;

CREATE INDEX job_status ON job(status);
