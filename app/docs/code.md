# Code
How to write Python for run_code: `babasu` (projects, images and their files), the api client, and worked examples.

Use run_code for what the API doesn't do in one call: going through many images, measuring and comparing, working on pixels (edited copies, crops, measurements), a change no operation makes. For one read or change, read and change are simpler; for changing many images that match a filter, POST /api/projects/ID/changes.

## babasu
- `babasu.project(id)`, `babasu.projects()`, `babasu.new_project(name)`: a project has `id`, `name` and `classes` (their names).
- `project.images(where="")`: its images (those matching a filter, read docs: filters), each an `Image` with `media_id`, `filename`, `width`, `height`, `class_counts`, `annotations` (its objects), `is_video` and `file`.
- `image.file` is the path of its file, the original as it came, a video's too. Read it with pixelflow: `pf.read_image(image.file)` is an RGB array; `pf.VideoReader(image.file)` gives a video's frames (RGB) and its `fps`. Never write to it.
- `project.add(*paths)`: puts image and video files (or folders of them) in as new images, and waits until they're in: `{added, already_there, skipped}`. Make the files in your folder (`pf.save_image`, `pf.VideoWriter`), then add them.
- `image.set_annotations(objects)`: replaces its objects with the whole list (leave out what goes).
- `project.find(["car", "person"], where="")` finds and outlines classes on its images, as the find tool does; `project.run(workflow_id, where="")` runs a saved workflow. Both wait until the job is done and answer its result.
- `babasu.wait(job_id)` waits for any job; one that fails raises `Refused`.

## The client
- `api.get(path)`, `api.post(path, body)`, `api.put(path, body)`, `api.patch(path, body)`, `api.delete(path)`: any /api operation (find_api finds one and what it takes). Each returns the JSON answer; a write's says what it did (`did`).
- A refusal raises `Refused` with the API's reason: catch it to go on with the others, or let it stop the code.
- `print()` what you need to see and what you'll report: the output comes back to you (8,000 characters at most).
- The code's changes are part of your turn: undoing the turn (History) undoes them.
- Read any file; change projects, images and annotations only through `babasu` and `api`, never the database or the app's own files.
- pixelflow (`import pixelflow as pf`; read_docs pixelflow: its docs) and numpy are there. A run stops after 60 seconds unless you give more (`seconds`, up to 600).
- `progress(done, total)` shows a bar in the chat while a long loop runs; what you print shows there as it's printed.

## Your folder
Each chat has a folder of its own, and your code runs in it: files open by relative paths. The functions and variables a run makes are there for the next runs. Keep there what a step finds (`json.dump(misses, open("results/misses.json", "w"))`) and drafts (workflows) before they're saved; read them back the same way.

The folder is yours, not the app's: projects, images, annotations and runs change only through `babasu` and `api`, so the app shows them and History can undo them.

## Work in steps
Your runs in a chat share one Python process, so what a run leaves in variables the next run has; files stay for good (variables go when the process stops). A long request is done in steps, each on what the last one left:

1. Read and compute: save what you'll need (a variable, and a file in `results/`), and print what you need to decide and the checks (how many read, matched, left out), never the data itself.
2. Look and decide: look at a few of them, think, choose the next step.
3. Go on with what the last step saved: no copying ids or numbers out of what was printed.

A run that's stopped (time) or crashes ends the process, and so do 30 minutes without a run: its variables are gone; read the files again and go on.

## A project's images
`babasu.project(project_id).images()` is a list of them all; `.images("car")` of those matching a filter (read docs: filters). It's read whole before it's returned, so changing the images while going through it is safe.

An image's `annotations` are its objects, each with `class` (its name) and `bbox` [x, y, w, h] in pixels, `segments` (an outline) or `keypoints`, and `score` on a model's; a video's also have `frame` and `track`. Write them back with `image.set_annotations(objects)` and the whole list: leave out what goes, change what changes. A class name the project hasn't had becomes a class.

## Grey copies of a project's images and videos
The copies go into a new project; the originals stay as they are.

```python
import os
import pixelflow as pf

source = babasu.project(3)
os.makedirs("grey", exist_ok=True)
for image in source.images():
    made = f"grey/{image.filename}"
    if image.is_video:
        video = pf.VideoReader(image.file)
        with pf.VideoWriter(made, like=video) as writer:
            for frame in video:
                writer.write(pf.transforms.to_grayscale(frame, keep_channels=True))
    else:
        pf.save_image(made, pf.transforms.to_grayscale(pf.read_image(image.file), keep_channels=True))
grey = babasu.new_project(f"{source.name} in grey")
print(grey.add("grey"))
```

## Crop every person into a new project
```python
import os
import pixelflow as pf

source = babasu.project(3)
os.makedirs("people", exist_ok=True)
for image in source.images("person"):
    if image.is_video:
        continue  # a video's objects are on frames: read them with pf.VideoReader
    pixels = pf.read_image(image.file)
    for n, a in enumerate(a for a in image.annotations if a["class"] == "person"):
        x, y, w, h = (int(v) for v in a["bbox"])
        pf.save_image(f"people/{image.media_id}-{n}.png", pixels[y:y + h, x:x + w])
people = babasu.new_project("People")
print(people.add("people"))
```

## Delete one class's objects
```python
changed = removed = 0
for image in babasu.project(1).images("star"):
    kept = [a for a in image.annotations if a["class"] != "star"]
    image.set_annotations(kept)
    changed, removed = changed + 1, removed + len(image.annotations) - len(kept)
print(f"Removed {removed} star objects from {changed} images")
```
(`PATCH /api/projects/{id}/classes` with `{"remove": ["star"]}` does it in one call, and removes the class.)

## Combine projects into a new one
The images of two projects that have a circle go into a new project, each with only its circles. The files aren't copied: `POST /api/projects/{id}/images` copies a set of images in, with their objects; an image whose file the new project has already is copied once (`already_there` the second time).

```python
circles = babasu.new_project("Circles")
for project_id in (1, 6):
    print(api.post(f"/api/projects/{circles.id}/images", {"images": {"project_id": project_id, "where": "circle"}})["did"])
for image in circles.images():
    kept = [a for a in image.annotations if a["class"] == "circle"]
    if len(kept) < len(image.annotations):
        image.set_annotations(kept)
```
