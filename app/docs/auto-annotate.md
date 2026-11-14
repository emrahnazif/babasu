# Auto-annotate
How auto-annotators find objects from the class prompts and add them to a project: what runs, what's skipped, what's kept.

## Auto-annotators
An auto-annotator (the project's `annotator` setting) is a workflow: an image and the project's class prompts go in, objects come out. The built-in ones:
- **SAM 3**: boxes and outlines; the SAM License. Suggested for detection and segmentation.
- **Grounding DINO**: boxes; Apache-2.0.
- **Grounding DINO → EdgeTAM**: boxes, then outlines; Apache-2.0 (the permissive choice).

A project points at one (its settings). The models an auto-annotator uses must be downloaded first (Models page, or download_model); a run without them fails and says which.

## A run
- It runs over the project's images matching a filter (all of them, some picked at random, or the ones given, e.g. the open image), for all classes or some.
- **Once per class:** an image is auto-annotated once for each class. Running again only does what's new: a class added since, images added since. With **again**, the images are done again. The image page's button always does its image again.
- Objects are **added** next to the ones there, never replacing them. A new object overlapping one of its class almost entirely (90%) is already there, and left out.
- Objects scoring below the **minimum score** are left out, beyond the auto-annotator's own threshold.
- Each object keeps its **score**; it's removed when someone edits the object.
- It's a background job in the gpu lane: one at a time with workflow runs and model loading. Its results are a run (get_run).

## Try
Try runs the auto-annotator on 4 images of the filter and shows what it finds, saving nothing: to find a good prompt or threshold before running. A prompt that works is saved to its class.

## Prompts
A class's prompt is what's looked for. Short concrete phrases work best ("red car", "hole in the road surface"). When a class finds too little or the wrong things, a better prompt usually helps more than a lower score.
