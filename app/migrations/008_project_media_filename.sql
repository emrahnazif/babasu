-- Images are listed in file-name order (ties by media id): the project page's pages and the
-- image page's previous/next read this index instead of sorting the whole project.
CREATE INDEX project_media_filename ON project_media(project_id, filename, media_id);
