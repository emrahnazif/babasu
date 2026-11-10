-- Each conversation remembers which agent it talks to; each reply remembers who wrote it.
-- Agents and models live in settings.json (app/settings.py), so these are plain text, not keys:
-- a deleted agent falls back to the default one, and old replies keep their names.
ALTER TABLE conversation ADD COLUMN agent_id TEXT;   -- NULL: the default agent
ALTER TABLE message ADD COLUMN agent TEXT;           -- assistant replies: the agent's name then
ALTER TABLE message ADD COLUMN model TEXT;           -- and its model's name
