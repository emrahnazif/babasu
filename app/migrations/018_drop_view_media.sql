-- Agent-written views are gone, and with them the images they kept as used (013). Their history
-- entries go too: there's nothing left to undo them into.

DROP TABLE view_media;
DELETE FROM history WHERE tbl = 'view_media' OR (tbl = 'file' AND json_extract(key, '$.path') LIKE 'views/%');
DELETE FROM action WHERE NOT EXISTS (SELECT 1 FROM history WHERE action_id = action.id);
