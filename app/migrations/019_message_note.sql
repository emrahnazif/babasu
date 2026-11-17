-- What the person sees of a tool call (app/agent_tools.py): its label, what it did; its note, what
-- came out in a few words (also what the model reads of an earlier turn's result); and the images
-- it found, shown in its row: JSON {"project_id", "media_ids"}.

ALTER TABLE message ADD COLUMN label TEXT;
ALTER TABLE message ADD COLUMN note TEXT;
ALTER TABLE message ADD COLUMN images TEXT;
