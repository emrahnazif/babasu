-- What the model's server counted for a reply: the tokens it read (the conversation sent) and the
-- tokens it wrote; empty when the server doesn't say.

ALTER TABLE message ADD COLUMN tokens_in INTEGER;
ALTER TABLE message ADD COLUMN tokens_out INTEGER;
