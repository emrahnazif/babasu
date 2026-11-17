"""Try the agent: one message through its real turn, every step printed, on a throwaway copy of
the workspace (the real one is never touched).

    uv run python -m scripts.try_agent "How many images does Shapes have?"
    uv run python -m scripts.try_agent --agent gpt-oss:20b --page /projects/14 "What would a report show?"
    uv run python -m scripts.try_agent --agent gpt-oss:20b --bare "Tell me how to write jinja code?"

--agent: a saved agent, by its name or its model's name (default: the default agent).
--page: the page the message is written on, as the chat sends it.
--bare: the message alone to the agent's model, without babasu's prompt and tools (what the model
does on its own), to compare.
--instructions: other instructions for the agent, in the copy only (to try a change before making it).
--keep: keep the copy (its path is printed) to look at what the agent made.

The copy has the database, workflows and settings; media and model weights are cloned
(copy-on-write on macOS, so instant and taking no space), so nothing the agent does reaches the
real files. Job workers run, so the agent's runs and downloads happen.
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import platformdirs

REAL = Path(platformdirs.user_data_dir("babasu", appauthor=False))


def copy_workspace(home: Path) -> None:
    """The workspace under a new HOME: data copied, media and weights cloned."""
    target = Path(platformdirs.user_data_dir("babasu", appauthor=False).replace(str(Path.home()), str(home)))
    target.mkdir(parents=True)
    with sqlite3.connect(REAL / "babasu.db") as source, sqlite3.connect(target / "babasu.db") as copy:
        source.backup(copy)  # consistent even while the app has the database open
    for name in ("workflows", "jobs", "conversations", "trash", "settings.json"):
        path = REAL / name
        if path.is_dir():
            shutil.copytree(path, target / name)
        elif path.is_file():
            shutil.copy2(path, target / name)
    for name in ("media", "weights"):
        if (REAL / name).is_dir():
            clone = subprocess.run(["cp", "-cR", str(REAL / name), str(target / name)], capture_output=True, check=False)
            if clone.returncode:  # no clonefile here: a real copy
                shutil.copytree(REAL / name, target / name, dirs_exist_ok=True)


def pick_agent(settings, wanted: str | None):
    if not wanted:
        return settings.find_agent(None)
    s = settings.get()
    for agent in s.agents:
        model = settings.find_model(agent.model_id)
        if wanted in (agent.name, agent.id) or (model and wanted in (model.name, model.model)):
            return agent, model
    sys.exit(f"No agent or model named {wanted!r}")


def bare(settings, wanted: str | None, text: str) -> None:
    from app import agent as agent_mode

    _, model = pick_agent(settings, wanted)
    started = time.monotonic()
    with agent_mode.client(model) as client:
        answer = client.chat.completions.create(model=model.model, messages=[{"role": "user", "content": text}])
    message = answer.choices[0].message
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
    print(message.content)
    print(f"\n[{model.name}, bare: {time.monotonic() - started:.1f} s, {answer.usage.prompt_tokens} tokens in, "
          f"{answer.usage.completion_tokens} out, thinking {len(reasoning):,} characters, "
          f"answer {len(message.content or ''):,} characters]")


def short(value, size: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= size else text[:size] + f"… ({len(text):,} characters)"


def turn(settings, wanted: str | None, text: str, page: str | None, instructions: str | None) -> None:
    from contextlib import closing

    from app import agent as agent_mode
    from app import db, jobs, vision

    db.migrate()
    vision.setup()
    jobs.run_workers()
    chosen, model = pick_agent(settings, wanted)
    if instructions is not None:  # the copy's settings only
        settings.update(agents=[a.model_copy(update={"instructions": instructions}) if a.id == chosen.id else a
                                for a in settings.get().agents])
        chosen = chosen.model_copy(update={"instructions": instructions})
    with closing(db.connect()) as conn:
        conversation_id = agent_mode.new_conversation(conn)
        with conn:
            conn.execute("UPDATE conversation SET agent_id = ? WHERE id = ?", (chosen.id, conversation_id))

        def call_of(call_id: str) -> dict:
            for row in conn.execute("SELECT tool_calls FROM message WHERE conversation_id = ? AND tool_calls IS NOT NULL "
                                    "ORDER BY id DESC", (conversation_id,)):
                found = [c for c in json.loads(row[0]) if c["id"] == call_id]
                if found:
                    return found[0]
            return {}

        def result_of(call_id: str) -> str:
            row = conn.execute("SELECT content FROM message WHERE tool_call_id = ?", (call_id,)).fetchone()
            return row[0] if row else ""

        print(f"[{chosen.name} on {model.name}, conversation {conversation_id}, page {page or '-'}]\n")
        started = last = time.monotonic()
        writing = False
        for event in agent_mode.run_turn(conversation_id, text, threading.Event(), page, page):
            now = time.monotonic()
            kind = event["type"]
            if kind == "text":
                print(event["delta"], end="", flush=True)
                writing = True
                continue
            if writing:
                print()
                writing = False
            if kind == "tool" and event["status"] == "running":
                call = call_of(event["id"])
                print(f"\n→ {event['name']}  ({now - last:.1f} s thinking)  {short(call.get('arguments', ''), 600)}")
            elif kind == "tool":
                print(f"  {'✓' if event['status'] == 'done' else '✗'} {short(result_of(event['id']))}  ({now - last:.1f} s)")
            elif kind == "output":
                print(f"  | {event['text'].rstrip()}")
            elif kind in ("done", "stopped", "error"):
                print(f"\n[{kind}{': ' + event.get('message', '') if kind == 'error' else ''}; "
                      f"{now - started:.1f} s in all]")
            last = now if kind != "call" else last
    jobs.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("message")
    parser.add_argument("--agent")
    parser.add_argument("--page")
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--instructions")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="babasu-try-"))
    try:
        if not args.bare:  # --bare reads only the settings, so it needs no copy
            copy_workspace(home)
            os.environ["HOME"] = str(home)  # before the app is imported: its DATA_DIR is the copy
        import app.server  # noqa: F401  (the API the agent's tools call)
        from app import settings

        if args.bare:
            bare(settings, args.agent, args.message)
        else:
            turn(settings, args.agent, args.message, args.page, args.instructions)
    finally:
        if args.keep and not args.bare:
            print(f"\nThe copy: {home}")
        else:
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
