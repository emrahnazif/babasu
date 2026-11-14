-- What was done to the user's data in the last 30 days, so any of it can be undone (app/history.py).
-- An action is one request, job or agent turn; its entries are what it wrote, kept as undoing it
-- needs: an insert keeps the row's key, an update or a delete the row as it was. The triggers
-- that write the entries are made by db.migrate from the tables' columns (db.HISTORY_TABLES).

CREATE TABLE action (
  id      TEXT PRIMARY KEY,                                -- made when it starts (db.action)
  source  TEXT NOT NULL CHECK (source IN ('request', 'job', 'agent', 'app')),
  label   TEXT NOT NULL,                                   -- "DELETE /api/projects/7", the job, the person's message; an undo: the label of what it undid
  undoes  TEXT REFERENCES action(id) ON DELETE SET NULL,  -- an undo: the action it undid
  at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))  -- its first write
) STRICT;

CREATE INDEX action_at ON action(at);
CREATE INDEX action_undoes ON action(undoes);

CREATE TABLE history (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,            -- the order things happened in
  action_id  TEXT NOT NULL REFERENCES action(id) ON DELETE CASCADE,
  tbl        TEXT NOT NULL,                                -- the table; 'file' for a file moved to the trash
  op         TEXT NOT NULL CHECK (op IN ('insert', 'update', 'delete', 'trash', 'untrash')),
  key        TEXT NOT NULL,                                -- JSON: the row's primary key; a file: {"path"}
  before     TEXT                                          -- JSON: the row as it was; a file: {"trash"}, where it is in the trash
) STRICT;

CREATE INDEX history_action ON history(action_id, tbl, key);
CREATE INDEX history_row ON history(tbl, key);
