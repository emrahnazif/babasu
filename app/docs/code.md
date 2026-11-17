# Code
How to write Python for run_code: the api client, paging through a project's images, and worked examples.

Use run_code for what the API doesn't do in one call: going through many images, measuring and comparing, a change no operation makes. For one read or change, read and change are simpler; for changing many images that match a filter, POST /api/projects/ID/changes.

## The client
- `api.get(path)`, `api.post(path, body)`, `api.put(path, body)`, `api.patch(path, body)`, `api.delete(path)`: any /api operation (find_api finds one and what it takes). Each returns the JSON answer; a write's says what it did (`did`). find_api shows what an operation takes and the fields of its answer, so look there before writing code that reads one.
- `api.images(project_id, where="")`: a project's images (those matching a filter), each with its objects, a page at a time.
- A refusal raises `Refused` with the API's reason: catch it to go on with the others, or let it stop the code.
- `print()` what you need to see and what you'll report: the output comes back to you (8,000 characters at most).
- The code's changes are part of your turn: undoing the turn (History) undoes them.
- Work through the API only: no database, no files of the workspace.
- pixelflow and numpy are there. A run stops after 60 seconds unless you give more (`seconds`, up to 600).
- `progress(done, total)` shows a bar in the chat while a long loop runs; what you print shows there as it's printed.

## Your folder
Each chat has a folder of its own, and your code runs in it: files open by relative paths. The functions and variables a run makes are there for the next runs. Keep there what a step finds (`json.dump(misses, open("results/misses.json", "w"))`) and drafts (workflows) before they're saved; read them back the same way.

The folder is yours, not the app's: projects, images, annotations and runs change only through `api`, so the app shows them and History can undo them.

## Work in steps
Your runs in a chat share one Python process, so what a run leaves in variables the next run has; files stay for good (variables go when the process stops). A long request is done in steps, each on what the last one left:

1. Read and compute: save what you'll need (a variable, and a file in `results/`), and print what you need to decide and the checks (how many read, matched, left out), never the data itself.
2. Look and decide: look at a few of them, think, choose the next step.
3. Go on with what the last step saved: no copying ids or numbers out of what was printed.

A run that's stopped (time) or crashes ends the process, and so do 30 minutes without a run: its variables are gone; read the files again and go on.

## A project's images
`api.images(project_id)` goes through them all; `api.images(project_id, "car")` through those matching a filter (read docs: filters). Don't go through a filter while changing what it matches: it reads a page at a time (`GET /api/projects/{id}?page=N`), and the pages move under you. Go through all of them instead and pick in the code.

An image from `api.images` has `media_id`, `filename`, `width`, `height`, `objects` (how many of each class) and `annotations`: objects with `class_id` and `bbox` [x, y, w, h] in pixels, `segments` (an outline) or `keypoints`, and `score` on a model's. Write an image's objects back with `PUT /api/projects/{id}/images/{media_id}/annotations` and the whole list: leave out what goes, change what changes.

A class is known by its id in its project: read them with `GET /api/projects/{id}/classes` (id, name, prompt, color). Adding or changing classes (`PATCH` …/classes) answers them too, with the new ones' ids.

## Delete one class's objects
```python
project_id = 1
classes = api.get(f"/api/projects/{project_id}/classes")
star = next(c["id"] for c in classes if c["name"] == "star")
changed = removed = 0
for image in api.images(project_id):
    kept = [a for a in image["annotations"] if a["class_id"] != star]
    if len(kept) < len(image["annotations"]):
        api.put(f"/api/projects/{project_id}/images/{image['media_id']}/annotations", kept)
        changed, removed = changed + 1, removed + len(image["annotations"]) - len(kept)
print(f"Removed {removed} star objects from {changed} images")
```

## Merge one class into another
The objects of triangle become squares, then the triangle class goes.

```python
project_id = 1
classes = {c["name"]: c["id"] for c in api.get(f"/api/projects/{project_id}/classes")}
source, target = classes["triangle"], classes["square"]
moved = 0
for image in api.images(project_id):
    if any(a["class_id"] == source for a in image["annotations"]):
        objects = [a | {"class_id": target} if a["class_id"] == source else a for a in image["annotations"]]
        api.put(f"/api/projects/{project_id}/images/{image['media_id']}/annotations", objects)
        moved += sum(a["class_id"] == source for a in image["annotations"])
api.patch(f"/api/projects/{project_id}/classes", {"remove": [source]})
print(f"Moved {moved} triangle objects to square, and removed the triangle class")
```

## Combine projects into a new one
The images of two projects that have a circle go into a new project, each with only its circles. Images aren't copied: projects share the stored file (`POST /api/projects/{id}/images` adds a set of them, each under its name). An image in both projects is added once (`already_there` the second time).

```python
created = api.post("/api/projects", {"name": "Circles", "task": "detection", "classes": [{"name": "circle"}]})
new_id, circle_here = created["id"], created["classes"][0]["id"]
for project_id in (1, 6):
    print(api.post(f"/api/projects/{new_id}/images", {"images": {"project_id": project_id, "where": "circle"}})["did"])
    circle = {c["name"]: c["id"] for c in api.get(f"/api/projects/{project_id}/classes")}["circle"]
    for image in api.images(project_id, "circle"):  # only reading: the filter is safe here
        circles = [a | {"class_id": circle_here} for a in image["annotations"] if a["class_id"] == circle]
        api.put(f"/api/projects/{new_id}/images/{image['media_id']}/annotations", circles)
```
