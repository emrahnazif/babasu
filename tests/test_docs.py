import json
import re

import pytest

from app.agent_tools import DOCS, DocsArgs, ToolError, search_docs


def read(query: str) -> str | list[dict]:
    return search_docs(None, DocsArgs(query=query))


def test_every_page_in_pixelflows_index_can_be_read():
    """The agent reads pixelflow's pages by the paths its index (llms.txt) gives."""
    paths = re.findall(r"\]\(([^)]+)\.md\)", DOCS["pixelflow"])
    assert paths
    for path in paths:
        assert read(f"pixelflow/{path}").startswith("# ")


def test_a_link_inside_a_pixelflow_page_reads_that_page():
    """Pages link to each other as /docs/PATH.md; read as written, it's the pixelflow page."""
    assert read("/docs/core-concepts/detections.md") == read("pixelflow/core-concepts/detections")


def test_what_nothing_is_about_says_where_to_look(api):
    with pytest.raises(ToolError, match="or a topic: annotating"):
        read("xyzzy")


def _where(found: list[dict]) -> list[str]:
    return [f.get("operation") or f["doc"] + " " + f["text"].partition("\n")[0] for f in found]


@pytest.mark.parametrize("words, first", [
    ("rename class", "POST /api/projects/{project_id}/classes/rename"),  # models ask for the task, not a topic's name
    ("grey copies of videos", "code ## Grey copies of a project's images and videos"),
    ("crop people", "code ## Crop every person into a new project"),
    ("import files from a folder", "POST /api/projects/{project_id}/import"),
])
def test_words_find_the_part_of_the_docs_or_the_operation_about_them(api, words, first):
    found = read(words)
    assert _where(found)[0] == first, _where(found)
    assert len(found) <= 3


@pytest.mark.parametrize("words, found", [
    ("post /api/projects/222/changes", ["POST /api/projects/{project_id}/changes"]),
    ("patch /api/projects/{id}/classes hide", ["PATCH /api/projects/{project_id}/classes"]),
    ("check workflow post /api/workflows/check", ["POST /api/workflows/check"]),
    ("workflows/nodes", ["GET /api/workflows/nodes"]),
    ("delete /api/projects/119/changes", ["POST /api/projects/{project_id}/changes"]),  # not its method: all at the path
    ("projects/{project_id}", ["GET /api/projects/{project_id}", "PATCH /api/projects/{project_id}", "DELETE /api/projects/{project_id}"]),
])
def test_a_path_answers_its_operations_the_one_its_method_names_or_all(api, words, found):
    assert _where(read(words)) == found


def test_words_find_pixelflows_docs_too(api):
    assert any(f.get("doc", "").startswith("pixelflow/") for f in read("detections zones tracking"))


def test_an_operation_is_answered_whole_and_alone(api):
    found, = read("patch /api/projects/{project_id}")
    assert found["operation"] == "PATCH /api/projects/{project_id}" and found["parameters"]


def test_the_calls_the_api_page_gives_work(api):
    """search_docs api is where the agent learns the calls: each of its examples is answered."""
    project = api.post("/api/projects", json={"name": "Street", "classes": [{"name": "car"}, {"name": "circle"},
                                                                           {"name": "triangle"}]}).json()["id"]
    for method, path, body in [
        ("get", "/api/search?q=street", None),
        ("get", "/api/projects?limit=10", None),
        ("get", f"/api/projects/{project}?where=(car w:<80)&annotations=1&page=1", None),
        ("get", f"/api/images?where=car&project_id={project}", None),
        ("get", f"/api/projects/{project}/stats?field=class", None),
        ("get", "/api/workflows/nodes?names=grounding_dino,save_to_babasu", None),
        ("post", f"/api/projects/{project}/changes", {"action": "delete_objects", "where": "(circle)"}),
        ("post", f"/api/projects/{project}/classes/rename", {"name": "triangle", "new_name": "pyramid"}),
        ("post", f"/api/projects/{project}/classes/clear", {"name": "circle"}),
    ]:
        answer = api.request(method, path, json=body)
        assert answer.status_code == 200, (path, answer.text)
    assert "api" in DOCS and "/api/search?q=" in DOCS["api"]


def test_the_workflow_the_docs_build_loads(api):
    """search_docs workflows starts from a whole document: it loads as written."""
    example = json.loads(re.search(r"```json\n(.*?)```", DOCS["workflows"], re.S)[1])
    answer = api.post("/api/workflows/check", json=example)
    assert answer.status_code == 200, answer.text
