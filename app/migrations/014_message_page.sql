-- Each message of the person's keeps the page it was written on again (007 kept only the
-- conversation's last one, for reopening it there): the agent is told it with the message, and
-- "this" in an earlier message means the page it was written on.
ALTER TABLE message ADD COLUMN page TEXT;        -- user messages: the path, e.g. /projects/3
ALTER TABLE message ADD COLUMN page_title TEXT;  -- and the page's title then, e.g. "COCO val"
