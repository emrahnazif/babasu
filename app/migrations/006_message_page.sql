-- The page the person was on when they sent a message, so a chat opened from the history goes
-- back to where it was left (and the agent can later know what they're looking at).
ALTER TABLE message ADD COLUMN page TEXT;        -- user messages: the path, e.g. /projects/3
ALTER TABLE message ADD COLUMN page_title TEXT;  -- and its title, e.g. "COCO val"
