-- The picture a tool call gave the model (a JPEG, e.g. a crop of an object it looked at): shown in
-- the call's step, and sent to the model with the turn it was taken in.

ALTER TABLE message ADD COLUMN picture BLOB;
