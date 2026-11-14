# The app
What babasu is, what it does today and what comes later, and its pages.

babasu is a local computer vision app: everything runs on the person's computer, and their images stay there. A person makes projects (datasets), imports images, annotates them (by hand with a click tool, or with vision models from the class names), filters and measures them, runs workflows over them, and has pages built for what they need (views). You, the agent, work through the same functions as the app's own pages.

## What it does today
- Projects with classes; importing images from files and folders (local disk or a network drive).
- Annotating one image in the editor: changing objects by hand, and the click tool that outlines an object from a click or a box (see annotating); auto-annotating with an auto-annotator (see auto-annotate).
- Filtering a project's images by what's in them, and measuring them (see filters).
- Workflows: mozo graphs of models and steps, built in the builder, run over a project's images (see workflows).
- Views: pages written by an agent, from reports to a workflow's Studio page (see views).
- Vision models: downloaded, loaded and unloaded on the Models page (see models).
- Background jobs for everything long (see jobs).
- History: everything done in the last 30 days (by the person, you and jobs), each request, job or agent turn undoable as one, deleted files included.

## Finding what's in images nobody annotated
The annotations say only what someone (or auto-annotate) marked. To find things in images that aren't marked ("our images with a person"), look with a model: run a detector workflow over them (POST /api/runs: e.g. grounding-dino with prompts ["person"], on a project's images or any media_ids), wait for its job, then read what it found per image (GET /api/runs/ID/annotations; each found object has a score). Auto-annotate does the same for a project's own classes and keeps what it finds.

## What comes later (not there yet)
Training models (mayaku), evaluation, exporting models, COCO import and export of annotations, classification projects, a review queue for model-made objects, video, versions as their own feature, and searching images by what's in them. Asked for one of these, say it isn't in the app yet.

## The pages
- /projects: every project. /projects/new: create one.
- /projects/ID: a project's images, with the filter as chips (?where=). /projects/ID/edit: its settings (name, description, type, auto-annotator, classes), Duplicate and Delete.
- /projects/ID/images/IMAGE_ID: one image in the editor, with its objects and the click tool; stepping to the previous and next image.
- /workflows: every workflow. /workflows/ID/edit: the builder. /studio/ID: a workflow's Studio page.
- /models: vision models. /jobs/ID: a job and its log. /views/ID: a view. /history: what was done in the last 30 days, with Undo.
- /settings/general, /settings/models, /settings/agents: the theme, and the language models and agents (you) run on.
