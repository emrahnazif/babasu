-- The app's schema (product.md "Data"). Later changes are numbered migrations after this one.

-- What someone is working on: its media, their annotations, and the jobs run on them. Nothing in it
-- is required to start: a drop makes one (product.md "Projects, media and annotations").
CREATE TABLE project (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused: a link or a message naming a deleted one never points at a newer one
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  classes     TEXT NOT NULL DEFAULT '[]',         -- JSON: preferences by class name [{"name", "prompt", "color"}], in order
  task        TEXT,                               -- projects.TASKS; NULL: not chosen
  annotator   TEXT,                               -- an id in annotators.FINDERS; NULL: the suggested one
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

-- An image in a project, with its own id. Its file is stored once by content (media.path), shared
-- by every row with the same sha256; a project has a file once.
CREATE TABLE media (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id  INTEGER NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  sha256      TEXT NOT NULL,                      -- of the stored file; also its name on disk
  ext         TEXT NOT NULL,
  width       INTEGER NOT NULL,
  height      INTEGER NOT NULL,
  name        TEXT NOT NULL,                      -- the file name it came with, e.g. train/0001.jpg
  split       TEXT,                               -- 'train', 'val', 'test' or NULL
  annotations TEXT NOT NULL DEFAULT '[]',         -- JSON list of projects.Annotation
  analysed    TEXT NOT NULL DEFAULT '{}',         -- JSON: workflow id -> the class names it looked for here (runs.pending)
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  UNIQUE (project_id, sha256)
) STRICT;

CREATE INDEX media_name ON media(project_id, name, id);  -- a project's images in file-name order
CREATE INDEX media_sha256 ON media(sha256);             -- whether a file is still used

-- One row per annotation, measured, so filters and statistics ask about objects ("a car between 8
-- and 16 px", "touching the edge") with indexes instead of reading every image's JSON.
-- media.annotations stays the source of truth: triggers rebuild an image's rows whenever its
-- annotations are written, so no write path can forget them.
CREATE TABLE annotation (
  media_id    INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
  idx         INTEGER NOT NULL,                -- its position in the image's annotations list
  class       TEXT NOT NULL COLLATE NOCASE,    -- "Tomato" is "tomato", in one project or across all
  x           REAL NOT NULL,                   -- box in pixels of the original image
  y           REAL NOT NULL,
  w           REAL NOT NULL,
  h           REAL NOT NULL,
  edge        INTEGER NOT NULL,                -- 1 if it touches the image's edge
  area        REAL GENERATED ALWAYS AS (w * h) VIRTUAL,
  aspect      REAL GENERATED ALWAYS AS (w / nullif(h, 0)) VIRTUAL,  -- width / height
  PRIMARY KEY (media_id, idx)
) STRICT, WITHOUT ROWID;

CREATE INDEX annotation_class ON annotation(class);

-- Each annotation of each image, measured: its box (one without a bbox gets the box around its
-- polygon or keypoints), and whether it touches the image's edge (within 1 px).
CREATE VIEW annotation_source AS
SELECT media_id, idx, class, x0 AS x, y0 AS y, x1 - x0 AS w, y1 - y0 AS h,
       (x0 <= 1 OR y0 <= 1 OR x1 >= image_width - 1 OR y1 >= image_height - 1) AS edge
FROM (
  SELECT m.id AS media_id, a.key AS idx, a.value ->> 'class' AS class, m.width AS image_width, m.height AS image_height,
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
  FROM media m, json_each(m.annotations) a
);

CREATE TRIGGER annotation_insert AFTER INSERT ON media BEGIN
  INSERT INTO annotation (media_id, idx, class, x, y, w, h, edge)
  SELECT media_id, idx, class, x, y, w, h, edge FROM annotation_source WHERE media_id = NEW.id;
END;

CREATE TRIGGER annotation_update AFTER UPDATE OF annotations ON media BEGIN
  DELETE FROM annotation WHERE media_id = OLD.id;
  INSERT INTO annotation (media_id, idx, class, x, y, w, h, edge)
  SELECT media_id, idx, class, x, y, w, h, edge FROM annotation_source WHERE media_id = NEW.id;
END;

-- Background jobs (imports, workflow runs, downloads…). See app/jobs.py and product.md "Jobs".
CREATE TABLE job (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused: logs are named by id
  kind        TEXT NOT NULL,
  project_id  INTEGER REFERENCES project(id) ON DELETE SET NULL,
  params      TEXT NOT NULL DEFAULT '{}',          -- JSON: the job function's arguments
  status      TEXT NOT NULL DEFAULT 'queued'
              CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted')),
  done        INTEGER NOT NULL DEFAULT 0,
  total       INTEGER,                             -- NULL until the job knows it
  message     TEXT NOT NULL DEFAULT '',            -- what it's doing now, e.g. the current file
  result      TEXT,                                -- JSON summary, when it finishes
  error       TEXT,
  stop_reason TEXT CHECK (stop_reason IN ('user', 'shutdown')),  -- set to ask a running job to stop
  waits_for   TEXT NOT NULL DEFAULT '[]',          -- JSON: job ids it waits for (e.g. a model's download)
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  started_at  TEXT,
  finished_at TEXT
) STRICT;

CREATE INDEX job_status ON job(status);

-- The agent chat: conversations and their messages, in the shape the chat completions API uses
-- (app/agent.py).
CREATE TABLE conversation (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  title      TEXT NOT NULL DEFAULT '',          -- the start of the first message
  agent_id   TEXT,                              -- settings.json's agent; NULL: the default one
  page       TEXT,                              -- where it was last open: the path, e.g. /projects/3
  page_title TEXT,                              -- and its name then
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE TABLE message (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
  content         TEXT NOT NULL DEFAULT '',     -- text; for a tool message, the tool's JSON result
  tool_calls      TEXT,                         -- assistant: JSON [{"id", "name", "arguments"}]
  tool_call_id    TEXT,                         -- tool: the call it answers
  status          TEXT NOT NULL DEFAULT 'done' CHECK (status IN ('streaming', 'done', 'error', 'stopped')),
  error           TEXT,
  agent           TEXT,                         -- assistant: the agent's name then
  model           TEXT,                         -- and its model's name
  page            TEXT,                         -- user: the page it was written on, e.g. /projects/3
  page_title      TEXT,                         -- and that page's title then
  about           TEXT,                         -- user: what was on that page, in its own words
  project_id      INTEGER REFERENCES project(id) ON DELETE SET NULL,  -- user: the project the images dropped with it made
  label           TEXT,                         -- tool: what the call did
  note            TEXT,                         -- tool: what came out, in a few words
  images          TEXT,                         -- tool: JSON media ids of the images it found
  picture         BLOB,                         -- tool: the picture it gave the model (JPEG)
  tokens_in       INTEGER,                      -- assistant: tokens the model's server read, if it says
  tokens_out      INTEGER,                      -- and wrote
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE INDEX message_conversation ON message(conversation_id, id);

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
