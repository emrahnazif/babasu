# Workflows and runs
Workflows are mozo graphs of models and steps; running one over images makes a run with its results.

## Workflows
- A workflow is a mozo graph: nodes (read an image, a model, crop, filter, draw…) joined output to input. GET /api/workflows/nodes lists the nodes (?names= gives them in full); each output connects to inputs of the same type.
- Built-in ones (SAM 3, Grounding DINO, Grounding DINO → EdgeTAM) can't be changed or deleted; editing one saves a copy.
- Built in the builder (/workflows/ID/edit), or written by you (below).
- A run keeps only what reaches a **save_to_babasu** node, under that node's label. A workflow can keep several results (e.g. two models side by side).
- The models it needs that aren't downloaded are downloaded first: the run waits for the download.

## Runs
- A run is a workflow run over images (a project's, or any by media id), as a background job in the gpu lane.
- It finds the prompts given, or the project's class prompts.
- Its results are per image and per label: the objects found, with class and score. GET /api/runs/ID gives them: per label and class, then per image.
- An auto-annotate run also adds its objects to the project; other runs only keep their results.
- Deleting a workflow leaves its runs: each keeps its own copy.

## Building one
1. The nodes it needs: GET /api/workflows/nodes, then ?names= for those in full.
2. Write it, then check it without saving: POST /api/workflows/check (what it keeps, its models, what's wrong).
3. Try it on 2–4 images before saving: POST /api/runs with `workflow` (the document) and a project's images (`count`); wait for its job.
4. Look at what it found: look with the run_id and an image (drawn), or read GET /api/runs/ID.
5. Fix and try again; then save it: POST /api/workflows (new) or PUT /api/workflows/ID.
