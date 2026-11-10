"""The agent: a conversation with a model through an OpenAI-compatible chat completions API.

The model can call babasu's own functions as tools (app/agent_tools.py). One turn: save the
person's message, ask the model, stream its text, run the tools it asks for, ask again, until it
answers without tools. Every step is saved as it happens, so the chat survives page changes and
shows what happened even after an error or a stop.
"""

import asyncio
import json
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import closing

import openai

from app import agent_tools, config, db, settings

MAX_ROUNDS = 8  # model calls per turn; each round may run tools
TITLE_LENGTH = 60

SYSTEM_PROMPT = f"""You are the assistant inside {config.APP_NAME}, a local computer vision app where people \
import images into projects, annotate them, train models and run workflows.
Answer from the app's real data: use the tools to look things up instead of guessing. If a tool fails, \
say so plainly. Keep answers short and concrete."""

_running: set[threading.Event] = set()  # the stop flag of each turn in progress


class NotConfigured(Exception):
    """No model is set up yet (Settings → Agent)."""


def configured() -> bool:
    s = settings.get()
    return bool(s.agent_base_url and s.agent_model)


def client() -> tuple[openai.OpenAI, str]:
    """A client for the configured model, and the model name. Raises NotConfigured."""
    if not configured():
        raise NotConfigured("Connect a model first: Settings → Agent.")
    s = settings.get()
    # Local servers accept any key, but the client requires one.
    return openai.OpenAI(base_url=s.agent_base_url, api_key=s.agent_api_key or "none", timeout=120), s.agent_model


def describe(exc: Exception) -> str:
    """A one-line explanation of a failed model call, for the chat and the settings page."""
    if isinstance(exc, NotConfigured):
        return str(exc)
    if isinstance(exc, openai.APIConnectionError):
        return f"Couldn't reach the model at {settings.get().agent_base_url}."
    if isinstance(exc, openai.AuthenticationError):
        return "The model's server refused the API key."
    if isinstance(exc, openai.NotFoundError):
        return f"The model {settings.get().agent_model!r} wasn't found on the server."
    if isinstance(exc, openai.APIStatusError):
        return f"The model's server answered {exc.status_code}: {exc.message}"
    return f"The model call failed: {exc}"


def test_connection() -> tuple[bool, str]:
    """Ask the configured model for a one-word reply: whether it worked, and a message to show."""
    try:
        model_client, model = client()
        with model_client:
            response = model_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with: ok"}], max_tokens=5
            )
    except Exception as exc:
        return False, describe(exc)
    return True, f"Connected: {model} replied {(response.choices[0].message.content or '').strip()!r}."


# --- Lifecycle: called by the server's lifespan ---


def recover() -> None:
    """At startup: a reply that was still streaming when the app stopped ends as stopped."""
    with closing(db.connect()) as conn, conn:
        conn.execute("UPDATE message SET status = 'stopped' WHERE status = 'streaming'")


def stop_all() -> None:
    """At shutdown: ask every turn in progress to stop; each saves what it has as stopped."""
    for stop in list(_running):
        stop.set()


# --- Conversations ---


def latest_conversation(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM conversation ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def new_conversation(conn: sqlite3.Connection) -> int:
    with conn:
        return conn.execute("INSERT INTO conversation DEFAULT VALUES").lastrowid


def _messages(conn: sqlite3.Connection, conversation_id: int | None, tool_results: bool = True) -> list[dict]:
    """The conversation's messages in order, tool calls parsed. tool_results=False leaves out
    the (possibly large) results, which only the model needs."""
    content = "content" if tool_results else "CASE role WHEN 'tool' THEN '' ELSE content END AS content"
    rows = conn.execute(
        f"SELECT role, {content}, tool_calls, tool_call_id, status, error FROM message "
        "WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return [dict(row) | {"tool_calls": json.loads(row["tool_calls"] or "[]")} for row in rows]


def turns(conn: sqlite3.Connection, conversation_id: int | None) -> list[dict]:
    """The conversation as the chat shows it: the person's turns, and each agent turn as one
    entry with its parts in order (text and tool calls) and how it ended."""
    messages = _messages(conn, conversation_id, tool_results=False)
    tool_status = {m["tool_call_id"]: m["status"] for m in messages if m["role"] == "tool"}
    result: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            result.append({"role": "user", "text": m["content"]})
        elif m["role"] == "assistant":
            if not result or result[-1]["role"] != "agent":
                result.append({"role": "agent", "parts": [], "status": "done", "error": None})
            turn = result[-1]
            if m["content"]:
                turn["parts"].append({"kind": "text", "text": m["content"]})
            for call in m["tool_calls"]:
                turn["parts"].append({"kind": "tool", "name": call["name"], "status": tool_status.get(call["id"], "error")})
            turn["status"], turn["error"] = m["status"], m["error"]
    return result


def _history(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    """The conversation as the model gets it. A tool call without a saved result (the app stopped
    in between) is left out, since the API requires every call to have its result."""
    messages = _messages(conn, conversation_id)
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        if m["role"] == "user":
            history.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            calls = [c for c in m["tool_calls"] if c["id"] in answered]
            if not m["content"] and not calls:
                continue
            message = {"role": "assistant", "content": m["content"]}
            if calls:
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ]
            history.append(message)
        else:
            history.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return history


# --- One turn ---


async def stream_turn(text: str) -> AsyncIterator[dict]:
    """Run one turn in its own thread and yield its events as they come (see run_turn).

    If the consumer goes away (the person pressed stop, or the app is closing: stop_all), the
    thread is told to stop at its next step, so the turn saves itself as stopped instead of
    waiting forever for someone to read it.
    """
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def run() -> None:
        with closing(db.connect()) as conn:
            conversation_id = latest_conversation(conn) or new_conversation(conn)
        turn = run_turn(conversation_id, text)
        try:
            for event in turn:
                loop.call_soon_threadsafe(events.put_nowait, event)
                if stop.is_set():
                    turn.close()  # GeneratorExit inside the turn: it saves what it has as stopped
                    break
        finally:
            _running.discard(stop)
            loop.call_soon_threadsafe(events.put_nowait, None)

    _running.add(stop)
    threading.Thread(target=run, name="agent-turn", daemon=True).start()
    try:
        while (event := await events.get()) is not None:
            yield event
    finally:
        stop.set()


def run_turn(conversation_id: int, text: str) -> Iterator[dict]:
    """Send the person's message and yield what happens, for the chat to show as it streams:
      {"type": "text", "delta": "..."}                      part of the reply
      {"type": "tool", "id", "name", "status": "running" | "done" | "error"}
      {"type": "done"} or {"type": "error", "message": "..."}
    If the generator is closed early (a stop), the reply so far is saved as stopped.
    """
    with closing(db.connect()) as conn:
        with conn:
            conn.execute("INSERT INTO message (conversation_id, role, content) VALUES (?, 'user', ?)", (conversation_id, text))
            conn.execute(
                f"UPDATE conversation SET updated_at = {db.NOW}, title = CASE title WHEN '' THEN ? ELSE title END WHERE id = ?",
                (text[:TITLE_LENGTH], conversation_id),
            )

        def fail(message: str) -> dict:
            """A turn that ends before any reply: an error message of its own."""
            with conn:
                conn.execute(
                    "INSERT INTO message (conversation_id, role, status, error) VALUES (?, 'assistant', 'error', ?)",
                    (conversation_id, message),
                )
            return {"type": "error", "message": message}

        try:
            model_client, model = client()
        except NotConfigured as exc:
            yield fail(str(exc))
            return

        with model_client:
            for _ in range(MAX_ROUNDS):
                reply_id = conn.execute(
                    "INSERT INTO message (conversation_id, role, status) VALUES (?, 'assistant', 'streaming')",
                    (conversation_id,),
                ).lastrowid
                conn.commit()
                text_parts: list[str] = []
                calls: dict[int, dict] = {}  # by index in the stream: {"id", "name", "arguments"}

                def finish(status: str, error: str | None = None, tool_calls: list | None = None) -> None:
                    with conn:
                        conn.execute(
                            "UPDATE message SET content = ?, tool_calls = ?, status = ?, error = ? WHERE id = ?",
                            ("".join(text_parts), json.dumps(tool_calls) if tool_calls else None, status, error, reply_id),
                        )

                try:
                    stream = model_client.chat.completions.create(
                        model=model, messages=_history(conn, conversation_id), tools=agent_tools.schemas(), stream=True
                    )
                    with stream:
                        for chunk in stream:
                            delta = chunk.choices[0].delta if chunk.choices else None
                            if delta is None:
                                continue
                            if delta.content:
                                text_parts.append(delta.content)
                                yield {"type": "text", "delta": delta.content}
                            for part in delta.tool_calls or []:
                                call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                                call["id"] = part.id or call["id"]
                                if part.function:
                                    call["name"] += part.function.name or ""
                                    call["arguments"] += part.function.arguments or ""
                except GeneratorExit:  # stopped: keep what arrived
                    finish("stopped")
                    raise
                except Exception as exc:  # unreachable server, bad key, unknown model…
                    finish("error", describe(exc))
                    yield {"type": "error", "message": describe(exc)}
                    return

                tool_calls = list(calls.values())
                finish("done", tool_calls=tool_calls)
                if not tool_calls:
                    yield {"type": "done"}
                    return

                for call in tool_calls:
                    yield {"type": "tool", "id": call["id"], "name": call["name"], "status": "running"}
                    try:
                        result, status, error = agent_tools.call(conn, call["name"], call["arguments"]), "done", None
                    except Exception as exc:  # the model sees the error and can correct itself
                        error = str(exc)
                        result, status = json.dumps({"error": error}), "error"
                    with conn:
                        conn.execute(
                            "INSERT INTO message (conversation_id, role, content, tool_call_id, status, error) "
                            "VALUES (?, 'tool', ?, ?, ?, ?)",
                            (conversation_id, result, call["id"], status, error),
                        )
                    yield {"type": "tool", "id": call["id"], "name": call["name"], "status": status}

        yield fail(f"Stopped after {MAX_ROUNDS} rounds of tool calls without an answer.")
