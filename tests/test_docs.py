import re

import pytest

from app.agent_tools import DOCS, DocsArgs, ToolError, read_docs


def read(topic: str) -> str:
    return read_docs(None, DocsArgs(topic=topic))


def test_every_page_in_pixelflows_index_can_be_read():
    """The agent reads pixelflow's pages by the paths its index (llms.txt) gives."""
    paths = re.findall(r"\]\(([^)]+)\.md\)", DOCS["pixelflow"])
    assert paths
    for path in paths:
        assert read(f"pixelflow/{path}").startswith("# ")


def test_a_link_inside_a_pixelflow_page_reads_that_page():
    """Pages link to each other as /docs/PATH.md; read as written, it's the pixelflow page."""
    assert read("/docs/core-concepts/detections.md") == read("pixelflow/core-concepts/detections")


@pytest.mark.parametrize("topic, says", [
    ("pixelflow/no-such-page", "read_docs pixelflow lists them"),
    ("no-such-topic", "the topics: annotating"),
])
def test_an_unknown_topic_says_where_to_look(topic, says):
    with pytest.raises(ToolError, match=says):
        read(topic)


def test_a_topic_that_isnt_one_answers_with_the_sections_about_it():
    """Models ask for the task ("rename class"), not a topic's name."""
    found = read("rename class")
    assert "From projects:\n## Classes" in found and "rename" in found


def test_the_calls_the_api_page_gives_work(api):
    """read_docs api is where the agent learns the calls: each of its examples is answered."""
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
