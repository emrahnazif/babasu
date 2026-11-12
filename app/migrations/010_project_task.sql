-- What each project is for and the annotator it uses, and the prompt of each class
-- (product.md "Annotation"). Existing projects become detection projects using SAM 3.

ALTER TABLE project ADD COLUMN task TEXT NOT NULL DEFAULT 'detection';  -- projects.TASKS
ALTER TABLE project ADD COLUMN annotator TEXT;  -- an id in annotators.FINDERS; NULL: none
UPDATE project SET annotator = 'sam3';

-- A class's prompt is the text sent to annotators: its name unless changed.
UPDATE project SET classes = (
  SELECT json_group_array(json_set(value, '$.prompt', coalesce(value ->> 'prompt', value ->> 'name')))
  FROM json_each(project.classes)
);
