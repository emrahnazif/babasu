# The app
What babasu is, what it does today and what comes later, and its pages.

babasu is a local computer vision app: everything runs on the person's computer, and their images stay there. A person makes projects (datasets), imports images, annotates them (by hand with a click tool, or with vision models from the class names), filters and measures them, and runs workflows over them. You, the agent, work through the same functions as the app's own pages.

## What it does today
- Asking about images without a project: dropped on Home (files or whole folders, any number), they go with the next chat message, which says how many and names them as an image set (see workflows).
- Projects with classes; importing images from files and folders (local disk or a network drive).
- Annotating one image in the editor: changing objects by hand, and the click tool that outlines an object from a click or a box (see annotating); auto-annotating with an auto-annotator (see auto-annotate).
- Filtering a project's images by what's in them, and measuring them (see filters).
- Workflows: mozo graphs of models and steps, built in the builder, run over a project's images (see workflows).
- Vision models: downloaded, loaded and unloaded on the Models page (see models).
- Background jobs for everything long (see jobs).
- History: everything done in the last 30 days (by the person, you and jobs), each request, job or agent turn undoable as one, deleted files included.

## Finding what's in images nobody annotated
The annotations say only what someone (or auto-annotate) marked. To find things in images that aren't marked ("our images with a person"), look with a model: run a workflow that finds them over the images (the run tool: images, e.g. those attached to a message or a project's, and prompts, e.g. ["person"]). It answers what was found per class and on how many images; GET /api/runs/ID has it per image (?annotations=1: the objects themselves, each with its score), and its page, /runs/ID, shows it: how many of each, and the images with what was found drawn. Auto-annotate does the same for a project's own classes and keeps what it finds.

## What comes later (not there yet)
Training models (mayaku), evaluation, exporting models, COCO import and export of annotations, classification projects, a review queue for model-made objects, video, versions as their own feature, and searching images by what's in them. Asked for one of these, say it isn't in the app yet.

## The pages
- /: Home, where the app opens: drop images and ask about them (find, count, describe); a way into making a project (a dataset).
- /projects: every project. /projects/new: create one.
- /projects/ID: a project's images, with the filter as chips (?where=). /projects/ID/edit: its settings (name, description, type, auto-annotator, classes), Duplicate and Delete.
- /projects/ID/images/IMAGE_ID: one image in the editor, with its objects and the click tool; stepping to the previous and next image.
- /runs/ID: a run: what it found per class, and its images with what it found drawn.
- /workflows: every workflow. /workflows/ID/edit: the builder. /studio: the workflows to use (a workflow's own Studio page isn't built yet).
- /models: vision models. /jobs/ID: a job and its log. /history: what was done in the last 30 days, with Undo.
- /settings/general, /settings/models, /settings/agents: the theme, and the language models and agents (you) run on.
