# Workflows and running them
Workflows are mozo graphs of models and steps; running one over images keeps what reaches its save nodes, in the projects.

## Workflows
- A workflow is a mozo graph: nodes (read an image, a model, crop, filter, draw…) joined output to input. GET /api/workflows/nodes lists the nodes (?names= gives them in full); each output connects to inputs of the same type.
- Built-in ones (SAM 3, Grounding DINO, Grounding DINO → EdgeTAM) can't be changed or deleted; editing one saves a copy.
- Built in the builder (/workflows/ID/edit), or written by you (below).
- A run keeps what reaches its save nodes, each what it says: **save_to_babasu** keeps detections as the input image's annotations, in its project; **save_image_to_babasu** keeps an image (a drawn, cropped or edited copy) as a new image, in the input's project or the one it names. Two save nodes keep twice: the workflow decides.
- The models it needs that aren't downloaded are downloaded first: the run waits for the download.

## Running one
- POST /api/annotate runs a workflow over images, as a background job in the gpu lane (the find tool starts one and waits). Its images, an image set: `{"project_id": 3, "where": "night", "count": 10}` (a project's, matching a filter, some at random), `{"where": "tomato"}` (every project's matching it) or `{"media_ids": [...]}`; images dropped in the chat are their project's. GET /api/images takes the same (?project_id=3) and says how many they are, with the first of them.
- With no workflow named, it runs the default one (the list marks it), which finds and outlines the classes named. Each class is looked for by its prompt in the image's project; a class the project hasn't had becomes one, looked for by its name.
- Each image remembers which classes a workflow looked for on it, so a run again does only what's new; with **again**, what that workflow's models found then (objects with a score) is replaced, and what people made or accepted stays.
- Its job's result says what it found per class. To see it: the project's images filtered by those classes (/projects/ID?where=person).
- Before running a model to find something, see whether the images are annotated with it already (GET /api/images?where=person): that may be the answer.

## Building one
1. The nodes it needs: GET /api/workflows/nodes, then ?names= for those in full.
2. Write it, then check it without saving: POST /api/workflows/check (what it keeps, its models, what's wrong).
3. Try it on 2–4 images before saving: POST /api/annotate with `workflow` (the document) and `images` (`{"project_id": 3, "count": 4}`); wait for its job. What it finds is kept, as any run's.
4. Look at what it found: look at an image (its objects, or one of them closely), or read GET /api/images/ID.
5. Fix and try again; then save it: POST /api/workflows (new) or PUT /api/workflows/ID.
