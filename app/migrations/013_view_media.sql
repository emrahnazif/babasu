-- The images a report shows (in the API answers it recorded), so the media cleanup keeps them
-- while the report exists (app/views.py).

CREATE TABLE view_media (
  view_id TEXT NOT NULL,
  media_id INTEGER NOT NULL REFERENCES media(id),
  PRIMARY KEY (view_id, media_id)
) STRICT, WITHOUT ROWID;

CREATE INDEX view_media_media_id ON view_media(media_id);
