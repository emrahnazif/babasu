# Auto-annotate
How auto-annotators find objects from the class prompts and add them to a project: what runs, what's skipped, what's kept.

## Auto-annotators
An auto-annotator (the project's `annotator` setting) is a workflow: an image and the project's class prompts go in, objects come out. The built-in ones:
- **SAM 3**: boxes and outlines; the SAM License. Suggested for detection and segmentation.
- **Grounding DINO**: boxes; Apache-2.0.
- **Grounding DINO → EdgeTAM**: boxes, then outlines; Apache-2.0 (the permissive choice).

A project may point at one (its settings); without one, the one suggested for its type is used. The models an auto-annotator uses that aren't downloaded are downloaded first, by itself: the run waits for the download (the dialog says which models and how big; so does downloads_first in what POST /api/annotate returns).

For one object at a time, as a person works, there's the click tool in the image editor (see annotating).

## A run
- It's a run of the auto-annotator (see workflows) over the project's images matching a filter (all of them, some picked at random, or the ones given, e.g. the open image), for all its classes or the ones named (by name; a name the project hasn't had yet becomes a class, found by its name).
- **Once per class:** an image is auto-annotated once for each class. Running again only does what's new: a class added since, images added since. With **again**, the images are done again, replacing what a model found on them before for those classes (objects with a score); what people made or accepted stays. The image page's button always does its image again.
- Otherwise objects are **added** next to the ones there.
- Objects scoring below the **minimum score** are left out, beyond the auto-annotator's own threshold.
- Each object keeps its **score**; it's removed when someone edits the object.
- It's a background job in the gpu lane: one at a time with other runs and model loading. Its result says what it found per class.

## Try
Try runs the auto-annotator on 4 images of the filter and shows them with what it found (kept, as any run's): to find a good prompt or threshold before running on all of them. A prompt that works is saved to its class.

## Prompts
A class's prompt is what's looked for. Short concrete phrases work best ("red car", "hole in the road surface"). When a class finds too little or the wrong things, a better prompt usually helps more than a lower score.
