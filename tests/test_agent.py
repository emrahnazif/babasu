"""How the agent talks to the model's server."""

import json

import numpy as np
import pixelflow as pf
import pytest

from app import agent, agent_tools, agents, media, projects, settings
from scripts.fake_model import Script, call, text, think, turn, use


def test_a_model_without_a_key_sends_the_placeholder_keyless_servers_accept():
    """Unsloth Studio's keyless access refuses an arbitrary bearer ("none") but takes "not-needed"."""
    model = settings.Model(id="local", name="Local", base_url="http://localhost:8888/v1", model="m")
    assert agent.client(model).api_key == "not-needed"
    assert agent.client(model.model_copy(update={"api_key": "sk-mine"})).api_key == "sk-mine"


@pytest.fixture
def model(conn):
    """model(rounds): the agent answering from a scripted model (scripts/fake_model.py); the script."""
    return lambda rounds: use(Script(rounds))


def test_what_the_model_thinks_streams_then_says_how_long_and_is_kept(model, conn):
    model([[think("The person "), think("wants a hello."), text("Hello!")]])
    events = turn("say hello")

    reasoning, = [e for e in events if e["type"] == "reasoning"]  # merged into one, as a later page replays it
    assert reasoning["delta"] == "The person wants a hello." and reasoning["since"] > 0
    kinds = [e["type"] for e in events]
    assert kinds.index("reasoning") < kinds.index("thought") < kinds.index("text")

    saved, = agent.turns(conn, agent.current_conversation(conn))[1:]
    thinking, said = saved["parts"]
    assert thinking["kind"] == "thinking" and thinking["text"] == "The person wants a hello."
    assert thinking["seconds"] == next(e for e in events if e["type"] == "thought")["seconds"]
    assert said == {"kind": "text", "text": "Hello!"}


def test_a_tool_call_shows_from_its_first_piece_with_what_it_does(model):
    """Its tool's own label from the first piece; the step the model wrote once it runs."""
    model([[call(0, "c1", "read", ""), call(0, "c1", "", '{"step": "Listing projects", "path": "/api/projects"}')],
           [text("None yet.")]])
    events = turn("any projects?")
    written, = [e for e in events if e["type"] == "call"]  # merged, as a later page replays it
    assert written["name"] == "read" and written["label"] == agent_tools.TOOLS["read"].label
    assert written["delta"] == '{"step": "Listing projects", "path": "/api/projects"}'
    assert next(e for e in events if e["type"] == "tool")["label"] == "Listing projects"


def test_a_saved_turn_has_its_thinking_text_and_tool_calls_in_order(model, conn):
    model([[think("Look first."), text("I'll look."), call(0, "c1", "read", '{"path": "/api/projects"}')],
           [think("Nothing there."), call(0, "c2", "read", '{"path": "/api/nowhere"}')],
           [text("No projects.")]])
    turn("any projects?")
    saved, = agent.turns(conn, agent.current_conversation(conn))[1:]
    assert [part["kind"] for part in saved["parts"]] == ["thinking", "text", "tool", "thinking", "tool", "text"]
    failed = saved["parts"][4]
    assert failed["status"] == "error" and failed["error"]


def test_a_models_thinking_is_sent_only_when_set(model):
    script = model([[text("Hi.")], [text("Hi again.")]])
    turn("hello")
    assert "reasoning_effort" not in script.requests[0]  # as its server does
    agents.update_model("fake", agents.ModelForm(name="Fake", base_url=settings.find_model("fake").base_url, model="fake", thinking="low"))
    turn("hello again")
    assert script.requests[1]["reasoning_effort"] == "low"


def test_the_forms_empty_thinking_leaves_it_to_the_server():
    assert agents.ModelForm(name="M", base_url="http://x/v1", model="m", thinking="").thinking is None


def test_look_sees_a_videos_frame_and_says_which(conn, video):
    with conn:
        project_id = projects.create(conn, "Clips")
    media_id = video(project_id)

    seen = agent_tools.look(conn, agent_tools.LookArgs(media_id=media_id, frame=3))
    assert seen.result == {"looked_at": "clip.mp4, frame 3"} and seen.picture
    assert agent_tools.look(conn, agent_tools.LookArgs(media_id=media_id)).result == {"looked_at": "clip.mp4, frame 0"}
    with pytest.raises(agent_tools.ToolError, match="frames 0 to 4"):
        agent_tools.look(conn, agent_tools.LookArgs(media_id=media_id, frame=9))


def test_a_turn_says_how_long_it_took_live_and_saved(model, conn):
    model([[text("Hi.")]])
    last = turn("hello")[-1]
    assert last["type"] == "done" and last["seconds"] >= 0
    saved, = agent.turns(conn, agent.current_conversation(conn))[1:]
    assert saved["seconds"] == last["seconds"]


def test_a_find_on_a_video_says_it_counted_frames():
    found = {"status": "succeeded", "objects_found": {"person": 10}, "looked_at_images": 1, "video_frames": 28}
    assert agent_tools._found_note(found) == "Found 10 person objects, looking at 1 image (28 video frames)"


def _recipe(title: str) -> str:
    """The code of a worked example in the code docs, by its heading."""
    doc = (agent_tools.config.DOCS_DIR / "code.md").read_text()
    section = doc.split(f"## {title}\n", 1)[1]
    return section.split("```python\n", 1)[1].split("```", 1)[0]


def test_code_makes_grey_copies_of_images_and_a_video_as_the_docs_show(model, conn, api, workers, image, video):
    """The worked example in docs/code.md, run as the agent's code: babasu reads each file, the copies come in."""
    with conn:
        project_id = projects.create(conn, "Street")
        for n, name in enumerate(["big.png", "empty.png", "small.png"]):
            media.store(conn, project_id, image(n * 100), name)  # each a different grey
    video(project_id)
    code = _recipe("Grey copies of a project's images and videos").replace("babasu.project(3)", f"babasu.project({project_id})")
    model([[call(0, "c1", "run_code", json.dumps({"code": code, "seconds": 120}))], [text("Done.")]])
    events = turn("make grey copies")

    ran = next(e for e in events if e["type"] == "tool" and e["status"] != "running")
    assert ran["status"] == "done", ran.get("error")
    grey, = [p for p in projects.list_all(conn) if p["name"] == "Street in grey"]
    copies = projects.detail(conn, grey["id"])["images"]
    assert sorted(c["filename"] for c in copies) == ["big.png", "clip.mp4", "empty.png", "small.png"]
    for copy in copies:
        found = projects.media_file(conn, copy["media_id"])
        pixels = next(iter(pf.VideoReader(found["path"]))) if found["video"] else pf.read_image(found["path"])
        assert np.ptp(pixels.astype(int), axis=-1).max() <= 2  # grey: red, green and blue the same


def test_code_imports_babasu_and_its_images_have_the_apis_names(model, api, street):
    """Code written as it would be anywhere imports what it uses; an image's names are the API's."""
    code = f"import babasu\nprint(sorted(i.filename for i in babasu.project({street}).images() if i.media_id))"
    model([[call(0, "c1", "run_code", json.dumps({"code": code}))], [text("Done.")]])
    ran = next(e for e in turn("list them") if e["type"] == "tool" and e["status"] != "running")
    assert ran["status"] == "done", ran.get("error")
    assert ran["note"] == "['big.png', 'empty.png', 'small.png']"


def test_code_crops_every_person_into_a_new_project_as_the_docs_show(model, conn, api, workers, street):
    code = _recipe("Crop every person into a new project").replace("babasu.project(3)", f"babasu.project({street})")
    model([[call(0, "c1", "run_code", json.dumps({"code": code}))], [text("Done.")]])
    events = turn("crop the people")

    ran = next(e for e in events if e["type"] == "tool" and e["status"] != "running")
    assert ran["status"] == "done", ran.get("error")
    people, = [p for p in projects.list_all(conn) if p["name"] == "People"]
    crop, = projects.detail(conn, people["id"])["images"]
    found = projects.image(conn, crop["media_id"])
    assert (found["width"], found["height"]) == (8, 20)  # the person's box in big.png


def test_a_message_with_dropped_images_goes_into_the_project_made_for_them(model, conn, workers, image, tmp_path):
    """As Continue does: the chat goes to the new project at once, and the message counts as written there."""
    (tmp_path / "a.png").write_bytes(image(1))
    script = model([[text("Two cars.")]])
    events = turn("count the cars", page="/", about="Home", paths=[str(tmp_path / "a.png")])

    into, = [e for e in events if e["type"] == "into"]
    kinds = [e["type"] for e in events]
    assert kinds.index("into") < kinds.index("attached") < kinds.index("agent")
    conversation_id = agent.current_conversation(conn)
    message = conn.execute("SELECT project_id, page, about FROM message WHERE role = 'user'").fetchone()
    assert into["url"] == message["page"] == f"/projects/{message['project_id']}" and message["about"] is None
    assert conn.execute("SELECT page FROM conversation WHERE id = ?", (conversation_id,)).fetchone()["page"] == into["url"]
    said = script.requests[0]["messages"][-1]["content"]
    assert into["url"] in said and "they're in it now" in said
