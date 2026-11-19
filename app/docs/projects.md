# Projects and classes
A project is what someone is working on: its images, their annotations and its classes; nothing in it is required. How classes, copying and deleting work.

## A project
- A **name**, a **description**, a **type** (task) and an **auto-annotator** (the `annotator` setting); all optional. Images dropped in the chat make a project of their own, named after them.
- The **type** is detection (boxes), segmentation (outlines) or keypoints (points); classification comes later. The type is a default, not a restriction: every project can hold every shape. It decides which auto-annotator is suggested. It can be changed any time; the annotations stay as they are.
- The **auto-annotator** is the workflow that auto-annotates it (see auto-annotate). Without one, it's the one suggested for its type (SAM 3 for detection, segmentation or no type; none for keypoints).

## Images
- Imported from files or folders, on the project's page, or dropped in the chat. Each image is in one project and has its own id (media_id; its page is /images/MEDIA_ID). The same file in two projects is two images, each with its own annotations; the file is stored once.
- An image's **annotations** (objects) each have a class (its name) and one or more shapes: a box, a polygon, keypoints. An object a model made has a **score** (its confidence) until someone edits it.
- An image is **labelled** when it has at least one object.

## Classes
- A class is a name on objects: annotating an object of a class the project hasn't had makes it one. Names are matched ignoring case, and kept as the project first spelt them.
- Each class has a **colour** and a **prompt**: the text auto-annotators look for. The prompt is the name unless changed, and a good prompt can find more ("hole in the road surface" for pothole).
- **Renaming a class** renames its objects; renaming it to another class's name merges them. **Removing a class** deletes its objects (History can undo it).

## Copying and moving images
POST /api/projects/ID/changes copies or moves the images matching a filter to another project, with their objects (the files aren't copied). POST /api/projects/ID/images copies a set of images in (see workflows: image sets). An image whose file the project has already is left as it is.

## Deleting
Deleting a project removes its images, their annotations and its runs. It can be undone for 30 days (see app: History). It's refused while one of its jobs is queued or running: cancel them first.
