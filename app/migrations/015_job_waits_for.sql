-- A job can wait for other jobs (e.g. a workflow run for the download of a model it uses): it
-- stays queued until they've succeeded, and ends the same way if one of them doesn't.
ALTER TABLE job ADD COLUMN waits_for TEXT NOT NULL DEFAULT '[]';  -- JSON: job ids
