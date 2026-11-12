-- A run made from a project's class prompts records which class each prompt stood for: a JSON
-- list of class ids, in prompt order (NULL: other prompts, or none). `annotates`: its results
-- became the project's annotations (auto-annotate), not only a look at them (Try).

ALTER TABLE run ADD COLUMN classes TEXT;
ALTER TABLE run ADD COLUMN annotates INTEGER NOT NULL DEFAULT 0 CHECK (annotates IN (0, 1));
