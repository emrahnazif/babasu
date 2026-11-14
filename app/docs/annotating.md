# Annotating
Working on one image in the editor: changing objects by hand, and the click tool that outlines an object from a click or a box, and its models (EdgeTAM, SAM 2, SAM 3).

## The editor
The image page (/projects/ID/images/IMAGE_ID) shows the image with its objects in their classes' colours. Selecting an object lets it be moved, resized, reshaped or deleted; the class menu in the toolbar changes the selected object's class, or, with nothing selected, the class new objects get. Every change is saved at once. ← and → go to the previous and next image of the project's filter. A project without classes, or a frozen one, only shows its objects.

## The click tool
The icon next to the class menu opens its menu: the model, and Turn off / Turn on (S). The choice stays as left, for every project. It's a different job from the auto-annotator, which finds every object of the classes from their prompts, in bulk, as a job. The click tool finds the one object the person points at, as they work.
- **Point** at an object: its outline shows, dashed, before anything is added.
- **Click**: it's fixed as a candidate, with Add (✓) and Discard (✕) next to it.
- **More clicks** shape it: inside it leaves that part out, outside it adds to it. Option/Alt-click does the opposite. Backspace takes back the last click.
- **Drag** a box: the object inside it. Clicking outside the box afterwards lets the object go beyond it.
- **Enter** or Add keeps it as an object of the current class; **Esc** or Discard drops it.
- **Refine** a selected object (R, or the button next to it): it becomes the candidate again, shaped the same way, and Add replaces it.

What it adds follows the project's type: a box in a detection project (an exact one, from the outline), the outline and its box in a segmentation project. An object kept or refined this way has no score: a person decided it.

## Its models
- **EdgeTAM** (the default): Apache-2.0, 56 MB. The lightest; good outlines.
- **SAM 2** (large): Apache-2.0, 898 MB. Better outlines on hard objects.
- **SAM 3**: the SAM License, 3.4 GB, about 5 GB of memory. The same model as the SAM 3 auto-annotator, so it's shared when both are used.

Each takes about the same time per point; a bigger one takes longer to prepare an image (EdgeTAM 0.05 s, SAM 2 0.5 s, SAM 3 1 s) and more memory. The image is prepared when it opens, and the next one while the person works. A model that isn't downloaded is downloaded when picked (its size shows in the menu); meanwhile a downloaded one answers, and the tool switches when the download ends. Loaded models show on the Models page.

For agents: get_click_tool and change_click_tool read and change the tool (on, model); the model in use answers `POST /api/projects/ID/images/IMAGE_ID/segment` (points with labels, 1 to include and 0 to leave out, and/or a box; the class), returning the object as an annotation without saving it.
