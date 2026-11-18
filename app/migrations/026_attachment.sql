-- The images attached to a person's chat message, as many as they dropped, in the order they came
-- (rowid). A message's `images` now names only the images a step of the agent found.

CREATE TABLE attachment (
  message_id INTEGER NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  media_id   INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
  PRIMARY KEY (message_id, media_id)
) STRICT;

CREATE INDEX attachment_media ON attachment(media_id);
