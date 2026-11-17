-- A stored image's own name: the file name it came with (imported, or attached to a chat message),
-- for wherever it's shown without a project's name for it (a run's results, adding it to a project).
-- Images stored before keep the name a project gives them.

ALTER TABLE media ADD COLUMN name TEXT;

UPDATE media SET name = (SELECT filename FROM project_media WHERE media_id = media.id ORDER BY project_id LIMIT 1);
