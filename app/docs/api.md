# The API: what to call for what
The calls for what people ask most: finding projects and images, reading what's in them, changing them, running and building workflows, following jobs. search_docs has every operation in full.

Reading is GET (the read tool); writing is POST, PUT, PATCH and DELETE (the change tool, or api.post… in code). A list too long to read says how to ask for less.

## Finding something by its name
- GET /api/search?q=street: the projects, images and workflows whose name has "street", each with its page. Use it whenever someone names something.
- GET /api/projects?limit=10: the projects most recently changed, with their image and class counts; &offset=10 for the next 10. Without limit, all of them.
- GET /api/workflows: every workflow; GET /api/jobs: the recent jobs.

## Reading a project and its images
- GET /api/projects/ID: the project, its classes with how many objects each, and its first images (100 a page: ?page=2).
- GET /api/projects/ID?where=FILTER: only the images matching a filter (search_docs filters), e.g. ?where=car, ?where=unlabelled, ?where=(car w:<80).
- GET /api/projects/ID?annotations=1: each image's objects too (boxes, outlines, scores).
- GET /api/images?where=car: how many images of every project have a car, and the first of them; ?project_id=ID for one project's.
- GET /api/projects/ID/stats?field=class: how many objects of each class; field=w, h, area or aspect: how their sizes spread.
- GET /api/images/MEDIA_ID: one image with its objects.

## Changing them
- Many images at once: POST /api/projects/ID/changes `{"action": "delete_objects", "where": "(circle)"}`. A preview first; the same with `"dry_run": false` does it (search_docs projects).
- Classes: POST /api/projects/ID/classes/rename `{"name": "triangle", "new_name": "pyramid"}`, …/classes/clear `{"name": "circle"}`, PATCH /api/projects/ID/classes (add, change, remove).
- One image's objects: PUT /api/images/MEDIA_ID/annotations with the whole list.
- A new project: POST /api/projects `{"name": "Circles"}`; its settings: PATCH /api/projects/ID.
- The pixels themselves (grey, resized, cropped or drawn copies, measurements): code reads each image's file and adds what it makes as new images (search_docs code); a workflow can make them too (search_docs workflows).

## Finding things with a model
- POST /api/annotate `{"images": {"project_id": 3, "count": 10}, "classes": ["person"]}` runs the default workflow over 10 of project 3's images; `"workflow_id": "ID"` runs another, saved one. It starts a job (the find tool does this and waits).

## Building a workflow
1. GET /api/workflows/nodes: the nodes there are; ?names=grounding_dino,save_to_babasu: their inputs, outputs and parameters.
2. POST /api/workflows/check with the document: what's wrong with it, without saving.
3. POST /api/workflows with the document: saves it, and answers its id.
4. POST /api/annotate with that `workflow_id` and a few images, to try it.
search_docs workflows has a whole document to start from.

## Jobs
GET /api/jobs/JOB_ID: a job's status, progress and result (the wait tool waits for it). POST /api/jobs/JOB_ID/cancel stops it.
