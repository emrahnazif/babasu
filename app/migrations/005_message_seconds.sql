-- How long a turn took, so the chat says it: on each reply, the turn's time when it ended (from
-- the person's message); the turn's last reply has the whole of it.
ALTER TABLE message ADD COLUMN seconds REAL;  -- assistant
