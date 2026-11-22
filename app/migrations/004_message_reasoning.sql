-- What a model thought before it replied (the reasoning its server streams beside the reply), kept
-- with the reply so the chat shows it as it did while it streamed: the text, and how long it took.
ALTER TABLE message ADD COLUMN reasoning TEXT;  -- assistant: what it thought, if its server sent it
ALTER TABLE message ADD COLUMN thought REAL;    -- and for how many seconds
