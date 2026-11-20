-- An action's undos and redos make a chain, a line (history.undo undoes a chain at its end), kept
-- on each action when it's made rather than followed through `undoes` on every read: the action
-- done that the chain starts from (`root`; itself for one done), and how many undos deep it is
-- (`depth`: 0 done, odd an undo, even a redo). The action an undo undid is the one a step less deep.

CREATE TABLE action_chain (
  id      TEXT PRIMARY KEY,                                -- made when it starts (db.action)
  source  TEXT NOT NULL CHECK (source IN ('request', 'job', 'agent', 'app')),
  label   TEXT NOT NULL,                                   -- "DELETE /api/projects/7", the job, the person's message; an undo: the label of what it undid
  root    TEXT NOT NULL,                                   -- the action done its chain starts from (it stays when that one is forgotten)
  depth   INTEGER NOT NULL DEFAULT 0,                      -- how many undos deep: 0 done, odd an undo, even a redo
  at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))  -- its first write
) STRICT;

WITH RECURSIVE chain(id, root, depth) AS (
  SELECT id, id, 0 FROM action WHERE undoes IS NULL
  UNION ALL SELECT a.id, root, depth + 1 FROM action a JOIN chain ON a.undoes = chain.id)
INSERT INTO action_chain (id, source, label, root, depth, at)
  SELECT a.id, source, label, root, depth, at FROM action a JOIN chain USING (id) ORDER BY a.rowid;

DROP TABLE action;
ALTER TABLE action_chain RENAME TO action;
CREATE INDEX action_at ON action(at);
CREATE INDEX action_root ON action(root, depth);
