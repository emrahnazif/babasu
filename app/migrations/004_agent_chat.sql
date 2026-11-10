-- The agent chat: conversations and their messages, in the shape the chat completions API uses.
-- See app/agent.py.

CREATE TABLE conversation (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  title      TEXT NOT NULL DEFAULT '',          -- the start of the first message
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
  status          TEXT NOT NULL DEFAULT 'done'
                  CHECK (status IN ('streaming', 'done', 'error', 'stopped')),
  error           TEXT,
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE INDEX message_conversation ON message(conversation_id, id);
