# Code
How to write Python for run_code: the api client, paging through a project's images, and worked examples.

Use run_code for what one API call doesn't do: going through many images, changing many objects at once, combining projects. For one change, call_api is simpler.

## The client
- `api.get(path)`, `api.post(path, body)`, `api.put(path, body)`, `api.patch(path, body)`, `api.delete(path)`: the operations in your instructions (find_api has their details). Each returns the JSON answer (None when there's none).
- `api.images(project_id, where="")`: a project's images (those matching a filter), each with its objects, a page at a time.
- A refusal raises `Refused` with the API's reason: catch it to go on with the others, or let it stop the code.
- `print()` what you need to see and what you'll report: the output comes back to you (8,000 characters at most).
- The code's changes are part of your turn: undoing the turn (History) undoes them.
- Work through the API only: no database, no files of the workspace.
- pixelflow and numpy are there. A run stops after 60 seconds unless you give more (`seconds`, up to 600).

## A project's images
`api.images(project_id)` goes through them all; `api.images(project_id, "car")` through those matching a filter (read docs: filters). Don't go through a filter while changing what it matches: it reads a page at a time (`GET /api/projects/{id}?page=N`), and the pages move under you. Go through all of them instead and pick in the code.

An image has `media_id`, `filename`, `width`, `height` and `annotations`: objects with `class_id` and `bbox` [x, y, w, h] in pixels, `segments` (an outline) or `keypoints`, and `score` on a model's. Write an image's objects back with `PUT /api/projects/{id}/images/{media_id}/annotations` and the whole list: leave out what goes, change what changes.

A class is known by its id in its project: read them with `GET /api/projects/{id}?images=0` (`classes`: id, name, prompt, color).

## Delete one class's objects
```python
project_id = 1
classes = api.get(f"/api/projects/{project_id}?images=0")["classes"]
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
classes = {c["name"]: c["id"] for c in api.get(f"/api/projects/{project_id}?images=0")["classes"]}
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
The images of two projects that have a circle go into a new project, each with only its circles. Images aren't copied: projects share the stored file (`POST /api/projects/{id}/images` adds one by its media_id). An image in both projects is added once (`added` is false the second time).

```python
new_id = api.post("/api/projects", {"name": "Circles", "task": "detection", "classes": [{"name": "circle"}]})["id"]
circle_here = api.get(f"/api/projects/{new_id}?images=0")["classes"][0]["id"]
added = 0
for project_id in (1, 6):
    classes = {c["name"]: c["id"] for c in api.get(f"/api/projects/{project_id}?images=0")["classes"]}
    circle = classes["circle"]
    for image in api.images(project_id, "circle"):  # only reading: the filter is safe here
        circles = [a | {"class_id": circle_here} for a in image["annotations"] if a["class_id"] == circle]
        answer = api.post(f"/api/projects/{new_id}/images",
                          {"media_id": image["media_id"], "filename": image["filename"], "annotations": circles})
        added += answer["added"]
print(f"Project {new_id}: {added} images with their circles")
```
