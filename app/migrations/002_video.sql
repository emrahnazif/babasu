-- Videos are media too (product.md "Video"): a row of `media` like an image, its original file
-- stored as it is. A row says what its file is (`type`, its MIME type); a video also its length,
-- frame count and frame rate, and the rate it's worked at (`sample_fps`: annotated, analysed and
-- shown every fps / sample_fps-th frame). An annotation on a video says its frame (the frame number
-- in the original file), so the annotation query table gets a `frame` column (NULL on an image).

ALTER TABLE media ADD COLUMN type TEXT NOT NULL DEFAULT '';
UPDATE media SET type = CASE ext WHEN 'jpg' THEN 'image/jpeg' WHEN 'png' THEN 'image/png' WHEN 'webp' THEN 'image/webp'
                                 WHEN 'bmp' THEN 'image/bmp' WHEN 'tif' THEN 'image/tiff' END;
ALTER TABLE media ADD COLUMN duration REAL;       -- a video's: seconds
ALTER TABLE media ADD COLUMN frame_count INTEGER; -- a video's: frames in the file
ALTER TABLE media ADD COLUMN fps REAL;            -- a video's own frame rate
ALTER TABLE media ADD COLUMN sample_fps REAL;     -- a video's: the rate it's worked at

DROP TRIGGER annotation_insert;
DROP TRIGGER annotation_update;
DROP VIEW annotation_source;
DROP TABLE annotation;

CREATE TABLE annotation (
  media_id    INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
  idx         INTEGER NOT NULL,                -- its position in the image's annotations list
  class       TEXT NOT NULL COLLATE NOCASE,    -- "Tomato" is "tomato", in one project or across all
  frame       INTEGER,                         -- a video's frame it's on; NULL on an image
  x           REAL NOT NULL,                   -- box in pixels of the original image
  y           REAL NOT NULL,
  w           REAL NOT NULL,
  h           REAL NOT NULL,
  edge        INTEGER NOT NULL,                -- 1 if it touches the image's edge
  area        REAL GENERATED ALWAYS AS (w * h) VIRTUAL,
  aspect      REAL GENERATED ALWAYS AS (w / nullif(h, 0)) VIRTUAL,  -- width / height
  PRIMARY KEY (media_id, idx)
) STRICT, WITHOUT ROWID;

CREATE INDEX annotation_class ON annotation(class);
CREATE INDEX annotation_frame ON annotation(media_id, frame);  -- a video's timeline

CREATE VIEW annotation_source AS
SELECT media_id, idx, class, frame, x0 AS x, y0 AS y, x1 - x0 AS w, y1 - y0 AS h,
       (x0 <= 1 OR y0 <= 1 OR x1 >= image_width - 1 OR y1 >= image_height - 1) AS edge
FROM (
  SELECT m.id AS media_id, a.key AS idx, a.value ->> 'class' AS class, a.value ->> 'frame' AS frame,
         m.width AS image_width, m.height AS image_height,
         coalesce(a.value -> 'bbox' ->> 0,
                  (SELECT min(p.value ->> 0) FROM json_each(a.value, '$.segments') p),
                  (SELECT min(k.value -> 'point' ->> 0) FROM json_each(a.value, '$.keypoints') k)) AS x0,
         coalesce(a.value -> 'bbox' ->> 1,
                  (SELECT min(p.value ->> 1) FROM json_each(a.value, '$.segments') p),
                  (SELECT min(k.value -> 'point' ->> 1) FROM json_each(a.value, '$.keypoints') k)) AS y0,
         coalesce((a.value -> 'bbox' ->> 0) + (a.value -> 'bbox' ->> 2),
                  (SELECT max(p.value ->> 0) FROM json_each(a.value, '$.segments') p),
                  (SELECT max(k.value -> 'point' ->> 0) FROM json_each(a.value, '$.keypoints') k)) AS x1,
         coalesce((a.value -> 'bbox' ->> 1) + (a.value -> 'bbox' ->> 3),
                  (SELECT max(p.value ->> 1) FROM json_each(a.value, '$.segments') p),
                  (SELECT max(k.value -> 'point' ->> 1) FROM json_each(a.value, '$.keypoints') k)) AS y1
  FROM media m, json_each(m.annotations) a
);

CREATE TRIGGER annotation_insert AFTER INSERT ON media BEGIN
  INSERT INTO annotation (media_id, idx, class, frame, x, y, w, h, edge)
  SELECT media_id, idx, class, frame, x, y, w, h, edge FROM annotation_source WHERE media_id = NEW.id;
END;

CREATE TRIGGER annotation_update AFTER UPDATE OF annotations ON media BEGIN
  DELETE FROM annotation WHERE media_id = OLD.id;
  INSERT INTO annotation (media_id, idx, class, frame, x, y, w, h, edge)
  SELECT media_id, idx, class, frame, x, y, w, h, edge FROM annotation_source WHERE media_id = NEW.id;
END;

INSERT INTO annotation (media_id, idx, class, frame, x, y, w, h, edge)
SELECT media_id, idx, class, frame, x, y, w, h, edge FROM annotation_source;
