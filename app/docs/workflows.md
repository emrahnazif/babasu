# Workflows and runs
Workflows are mozo graphs of models and steps; running one over images makes a run with its results; Studio pages show them.

## Workflows
- A workflow is a mozo graph: nodes (read an image, a model, crop, filter, draw…) joined output to input. list_nodes lists the nodes; each output connects to inputs of the same type.
- Built-in ones (SAM 3, Grounding DINO, Grounding DINO → EdgeTAM) can't be changed or deleted; editing one saves a copy.
- Built in the builder (/workflows/ID/edit), or written by you (validate_workflow, save_workflow).
- A run keeps only what reaches a **save_to_babasu** node, under that node's label. A workflow can keep several results (e.g. two models side by side).
- The models it needs that aren't downloaded are downloaded first: the run waits for the download.

## Runs
- A run is a workflow run over a project's images (or images dropped on a Studio page), as a background job in the gpu lane.
- It finds the prompts given, or the project's class prompts.
- Its results are per image and per label: the objects found, with class and score. get_run summarises them; /api/runs/ID/summary has them.
- An auto-annotate run also adds its objects to the project; other runs only keep their results.
- Deleting a workflow leaves its runs: each keeps its own copy.

## Studio
A workflow's Studio page (/studio/ID) is where its results get used: drop images in, see what it keeps. Its plain page shows the results; an agent can build it a page of its own (a studio view: counts, tables, charts) that replaces the plain one.
