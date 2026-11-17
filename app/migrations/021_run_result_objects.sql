-- What each run result found, per class: {"car": [how many, the sum of their scores]}, written with
-- the result (app/runs.py), so a run's totals and its pages don't read the results' masks.

ALTER TABLE run_result ADD COLUMN objects TEXT NOT NULL DEFAULT '{}';
UPDATE run_result SET objects = coalesce((
  SELECT json_group_object(name, json_array(n, s)) FROM (
    SELECT coalesce(nullif(json_extract(d.value, '$.class_name'), ''), 'class ' || json_extract(d.value, '$.class_id')) AS name,
           count(*) AS n, total(json_extract(d.value, '$.confidence')) AS s
    FROM json_each(run_result.result, '$.detections') d GROUP BY name)), '{}');
