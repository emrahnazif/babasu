"""What the API answers a call that's wrong, or written as people (and models) expect."""

import json

from app import config, db, projects


def test_a_listed_image_says_how_many_objects_of_each_class_it_has(api, street):
    listed = api.get("/api/images", params={"project_id": street, "where": "person"}).json()
    assert listed["count"] == 1 and listed["images"][0]["filename"] == "big.png"
    assert listed["images"][0]["class_counts"] == {"car": 1, "person": 1}


def test_a_missing_body_says_what_it_takes(api, street):
    refused = api.post(f"/api/projects/{street}/classes/clear").json()["detail"][0]["msg"]
    assert refused == "a JSON body is needed; it takes name, dry_run"


def test_removing_a_class_images_have_says_how_to_clear_it(api, street):
    refused = api.patch(f"/api/projects/{street}/classes", json={"remove": ["car"]}).json()["detail"]
    assert f'POST /api/projects/{street}/classes/clear {{"name": "car", "dry_run": false}}' in refused


def test_code_written_as_if_elsewhere_says_how_to_correct_it():
    from app.code import _hint

    tools = frozenset({"read"})
    assert "without an import" in _hint(ModuleNotFoundError("No module named 'api'", name="api"), tools)
    assert "api.get(path) reads" in _hint(NameError("name 'read' is not defined", name="read"), tools)


def test_the_project_list_can_be_asked_for_less(api, conn):
    """Many projects make a list too long to read: ?limit= and ?offset= page it, most recently changed first."""
    with db.action("request", "make"), conn:
        for name in ("First", "Second", "Third"):
            projects.create(conn, name)
    page = lambda **asked: [p["name"] for p in api.get("/api/projects", params=asked).json()]
    assert page(limit=1) == ["Third"] and page(limit=2, offset=1) == ["Second", "First"] and len(page()) == 3


def test_the_agent_reads_a_list_as_code_gets_it(api, conn):
    """One shape for one answer: reading /api/projects and api.get in code both give the list."""
    from app import agent_tools

    with db.action("request", "make"), conn:
        projects.create(conn, "Only")
    agent_tools._client = api
    assert agent_tools._answer("GET", "/api/projects") == api.get("/api/projects").json()


def test_an_images_file_name_is_called_the_same_everywhere(api, street):
    listed = api.get("/api/images", params={"project_id": street}).json()["images"][0]
    detail = api.get(f"/api/projects/{street}").json()["images"][0]
    assert "filename" in listed and "filename" in detail and "name" not in listed


def test_a_list_in_a_query_can_be_written_with_commas(api, street):
    ids = [i["media_id"] for i in api.get("/api/images", params={"project_id": street}).json()["images"]]
    assert api.get(f"/api/images?media_ids={ids[0]},{ids[1]}").json()["count"] == 2
    assert len(api.get("/api/workflows/nodes?names=crop,resize").json()) == 2


def test_a_body_field_that_isnt_one_is_named_as_json_writes_it():
    from app.routes import fields_at
    from app.routes.projects import ProjectChange

    assert "class" in fields_at(ProjectChange, ())


def test_a_node_giving_many_says_so(api):
    """An output that gives many (one crop per object) says what that means for the next node."""
    crop, = api.get("/api/workflows/nodes?names=crop_around_detections").json()
    assert crop["outputs"] == ["image: image (many: what it's wired to runs once for each)"]
    resize, = api.get("/api/workflows/nodes?names=resize").json()
    assert "many" not in resize["outputs"][0]


def test_a_settings_page_posts_to_its_own_address(api):
    """Its forms post to the settings address with ?next=, whatever page it was opened from."""
    import re
    html = api.get("/settings/models?next=/projects/new").text
    actions = [a for a in re.findall(r'<form[^>]*action="([^"]*)"', html) if a != "/agent/open"]
    assert actions == ["/settings/models?next=%2Fprojects%2Fnew"]



def test_a_workflow_is_saved_in_its_short_form(api):
    """Saved as written, without edges: an input written only where the latest earlier one isn't it."""
    document = {"name": "Square crops", "nodes": [
        {"id": "image", "type": "read_media"},
        {"id": "find", "type": "grounding_dino", "text": ["square"]},
        {"id": "draw", "type": "draw_boxes"},
        {"id": "crop", "type": "crop_around_detections", "inputs": {"image": "image"}},
        {"id": "keep", "type": "save_image_to_babasu"},
    ]}
    saved = api.post("/api/workflows", json=document).json()
    assert json.loads((config.WORKFLOWS_DIR / f"{saved['id']}.json").read_text()) == document
    assert api.get(f"/api/workflows/{saved['id']}").json()["graph"] == document
