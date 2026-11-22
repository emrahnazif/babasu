"""How the agent talks to the model's server."""

import pytest

from app import agent, agent_tools, agents, settings
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
    from app import projects

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
