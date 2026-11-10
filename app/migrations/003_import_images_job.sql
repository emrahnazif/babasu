-- The folder import job became "import_images": it takes a list of files and folders.
-- Earlier jobs keep their title and stay retryable.
UPDATE job
SET kind = 'import_images',
    params = json_object(
      'project_id', json_extract(params, '$.project_id'),
      'paths', json_array(json_extract(params, '$.path'))
    )
WHERE kind = 'import_folder';
