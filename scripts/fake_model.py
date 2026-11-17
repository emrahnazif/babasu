"""A scripted model, to run real agent turns without one: checking the chat, the turn and the
agent's tools as they are, live and saved (scripts/try_agent.py runs a real model).

A Script is what the model "answers", one list of deltas per model call (text, or tool calls), and
records each request it got. workspace() makes a throwaway workspace (HOME in a temp folder; with
copy=True the real workspace's data, see try_agent) whose agent runs on the script's server.

    from scripts.fake_model import Script, call, text, turn, workspace

    script = Script([[text("I'll look."), call(0, "c1", "search", '{"text": "car"}')], [text("Found it.")]])
    workspace(script)
    events = turn("find the car project")   # the turn's events, as the chat gets them
    script.requests[1]["messages"]          # what the model was sent the second time

Run as a script, it serves a throwaway workspace with a scripted turn in it, to look at in a browser:

    uv run python -m scripts.fake_model     # http://127.0.0.1:8765/projects
"""

import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path


def text(content: str) -> dict:
    return {"content": content}


def call(index: int, id: str, name: str, arguments: str) -> dict:
    return {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


class Script:
    """The model's answers, one list of deltas per call; `finish` and `delay` by call number: how a
    reply ends ("length": cut off) and how long the model waits before its first chunk."""

    def __init__(self, rounds: list[list[dict]], finish: dict[int, str] | None = None,
                 delay: dict[int, float] | None = None) -> None:
        self.rounds, self.finish, self.delay = rounds, finish or {}, delay or {}
        self.requests: list[dict] = []

    def chunks(self, n: int):
        time.sleep(self.delay.get(n, 0))
        deltas = self.rounds[n] if n < len(self.rounds) else [text("(the script has no more answers)")]
        reason = self.finish.get(n, "tool_calls" if any("tool_calls" in d for d in deltas) else "stop")
        for delta, end in [*((d, None) for d in deltas), ({}, reason)]:
            chunk = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": "fake",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": end}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.01)
        yield "data: [DONE]\n\n"


def _serve(script: Script) -> str:
    """The script as an OpenAI-compatible chat completions server; its base URL."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        script.requests.append(await request.json())
        return StreamingResponse(script.chunks(len(script.requests) - 1), media_type="text/event-stream")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}/v1"


def workspace(script: Script, copy: bool = False):
    """A throwaway workspace whose agent runs on the script; the app's server module. Call it
    before anything imports the app (its paths are set from HOME at import)."""
    home = Path(tempfile.mkdtemp(prefix="babasu-fake-"))
    if copy:
        from scripts.try_agent import copy_workspace
        copy_workspace(home)
    os.environ["HOME"] = str(home)
    from app import db, server, settings
    db.migrate()
    settings.update(models=[settings.Model(id="fake", name="Fake", base_url=_serve(script), model="fake")],
                    agents=[settings.Agent(id="fake", name="Assistant", model_id="fake")], default_agent="fake")
    return server


def turn(message: str, page: str | None = None) -> list[dict]:
    """Send a message as the chat does and wait for the turn to end: its events."""
    from app import agent
    running = agent.start_turn(message, page)
    while not running.ended:
        time.sleep(0.02)
    return running.events


if __name__ == "__main__":
    import uvicorn

    script = Script([
        [text("I'll check this project's labelled images."),
         call(0, "c1", "find_images", '{"step": "Finding labelled images", "project_id": 9, "where": "labelled"}'),
         call(1, "c2", "run_code", '{"step": "Counting objects per class", "code": "print(\\"car 41, person 12\\")"}')],
        [text("19 of the 24 images are labelled: **41 cars** and **12 people**.")],
    ])
    server = workspace(script, copy=True)
    turn("which images are labelled here, and what's in them?", "/projects/9")
    uvicorn.run(server.app, host="127.0.0.1", port=8765, log_level="warning")
