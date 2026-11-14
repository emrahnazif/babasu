# Projects and classes
A project is a dataset: images, their annotations, classes and a type; how classes, removing and duplicating work.

## A project
- A **name**, a **description**, a **type** (task) and an **auto-annotator** (the `annotator` setting).
- The **type** is detection (boxes), segmentation (outlines) or keypoints (points); classification comes later. The type is a default, not a restriction: every project can hold every shape. It decides which auto-annotator is suggested. It can be changed any time; the annotations stay as they are.
- The **auto-annotator** is the workflow that auto-annotates it (see auto-annotate). A new project gets the one suggested for its type (SAM 3 for detection and segmentation, none for keypoints).

## Images
- Imported from files or folders, on the project's page. An image is stored once by its content, however many projects use it; each project has its own file name for it and its own annotations.
- An image's **annotations** (objects) each have a class and one or more shapes: a box, a polygon, keypoints. An object a model made has a **score** (its confidence) until someone edits it.
- An image is **labelled** when it has at least one object.

## Classes
- Each class has an **id**, a **name**, a **colour** and a **prompt**: the text auto-annotators look for. The prompt is the name unless changed, and a good prompt can find more ("hole in the road surface" for pothole).
- Names are unique in a project, ignoring case. Ids are never reused.
- **Removing a class** doesn't delete its annotations: they're hidden (not shown, not counted, not in filters or statistics). Adding a class with that name again brings it back with its annotations, under its old id.

## Duplicating
A new project with the same description, type, annotator, classes (removed ones too), images and annotations; named "<name> copy" unless given a name. The image files are shared, so it takes no space. Runs stay with the original, so in the copy auto-annotate treats every image as not done yet.

## Frozen projects
A frozen project never changes; to change it, duplicate it. (Freezing comes with training.)

## Deleting
Deleting a project removes its images (from it), their annotations and its runs, and the image files no other project or run uses. It can be undone for 30 days (see app: History). It's refused while one of its jobs is queued or running: cancel them first.
