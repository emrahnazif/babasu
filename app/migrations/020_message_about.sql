-- What was on the page a person's message was written on, in the page's own words (its template's
-- `about` block, sent by the chat): the model reads it with the current message.

ALTER TABLE message ADD COLUMN about TEXT;
