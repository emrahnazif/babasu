-- Where each chat was last open: the page the person was on, kept up to date as they move around
-- with the chat open (not only when they write), so a chat reopens where it was left.
-- This replaces the page kept on each message (006); each chat starts from its last one.
ALTER TABLE conversation ADD COLUMN page TEXT;        -- the path, e.g. /projects/3
ALTER TABLE conversation ADD COLUMN page_title TEXT;  -- and its name then, e.g. "COCO val"

UPDATE conversation SET (page, page_title) = (
  SELECT page, page_title FROM message
  WHERE message.conversation_id = conversation.id AND page IS NOT NULL
  ORDER BY message.id DESC LIMIT 1
);

ALTER TABLE message DROP COLUMN page;
ALTER TABLE message DROP COLUMN page_title;
