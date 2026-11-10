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

# What every agent is told about the app; the agent's own instructions follow.
APP_PROMPT = f"""You are an agent inside {config.APP_NAME}, a local computer vision app where people \
import images into projects, annotate them, train models and run workflows. You can look at the app's data \
through your tools."""

_running: set[threading.Event] = set()  # the stop flag of each turn in progress


def client(model: settings.Model) -> openai.OpenAI:
    # Local servers accept any key, but the client requires one.
    return openai.OpenAI(base_url=model.base_url, api_key=model.api_key or "none", timeout=120)


def system_prompt(agent: settings.Agent) -> str:
    return f"{APP_PROMPT}\n\n{agent.instructions}".strip()


def describe(exc: Exception, model: settings.Model) -> str:
    """A one-line explanation of a failed model call, for the chat and the settings pages."""
    if isinstance(exc, openai.APIConnectionError):
        return f"Couldn't reach {model.name} at {model.base_url}. Is the server running?"
    if isinstance(exc, openai.AuthenticationError):
        return f"The server refused the API key of {model.name}."
    if isinstance(exc, openai.NotFoundError):
        # A wrong address and an unknown model both come back as 404.
        return (f"Nothing found at {model.base_url} for {model.model!r}: check the server URL and the model name. "
                f"{URL_HINT}")
    if isinstance(exc, openai.APIStatusError):
        return f"{model.name}: the server answered {exc.status_code}: {exc.message}"
    return f"{model.name}: the call failed: {exc}"


URL_HINT = "The URL is the server's OpenAI-compatible address, which usually ends in /v1 (Ollama: http://localhost:11434/v1)."


def test_connection(model: settings.Model) -> tuple[bool, str]:
    """Check the server, the model name and a one-word reply: whether it worked, and a message to show.

    The server's model list (if it has one) tells a wrong address apart from a wrong model name.
    """
    available: list[str] | None = None
    try:
        with client(model) as model_client:
            try:
                available = [m.id for m in model_client.models.list()]
            except openai.APIStatusError:
                available = None  # not every server lists its models; the reply below still tells
            if available and model.model not in available:
                return False, f"The server has no model {model.model!r}. It has: {', '.join(available[:15])}."
            response = model_client.chat.completions.create(
                model=model.model, messages=[{"role": "user", "content": "Reply with: ok"}], max_tokens=5
            )
    except openai.NotFoundError as exc:
        if available is None:  # neither a model list nor a reply: the address is wrong
            return False, f"No OpenAI-compatible API at {model.base_url}. {URL_HINT}"
        return False, describe(exc, model)
    except Exception as exc:
        return False, describe(exc, model)
    reply = (response.choices[0].message.content or "").strip()
    # A reasoning model may spend the few tokens allowed here thinking, and reply with nothing.
    return True, f"Connected: {model.model} replied {reply!r}." if reply else f"Connected: {model.model} answered."


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


def _exists(conn: sqlite3.Connection, conversation_id: int) -> bool:
    return conn.execute("SELECT 1 FROM conversation WHERE id = ?", (conversation_id,)).fetchone() is not None


def current_conversation(conn: sqlite3.Connection) -> int | None:
    """The conversation open in the chat; None for a new chat that hasn't started yet."""
    conversation_id = settings.get().conversation
    return conversation_id if conversation_id is not None and _exists(conn, conversation_id) else None


def open_conversation(conn: sqlite3.Connection, conversation_id: int | None) -> None:
    """Show this conversation in the chat; None starts a new chat (made by its first message)."""
    if conversation_id is not None and not _exists(conn, conversation_id):
        raise ValueError("There is no such conversation.")
    settings.update(conversation=conversation_id)


def new_conversation(conn: sqlite3.Connection) -> int:
    """Start a conversation and open it in the chat."""
    with conn:
        conversation_id = conn.execute("INSERT INTO conversation DEFAULT VALUES").lastrowid
    settings.update(conversation=conversation_id)
    return conversation_id


def conversations(conn: sqlite3.Connection, query: str = "", limit: int = 50) -> list[dict]:
    """Past conversations, most recently used first, each with the page it was last open on.
    `query` keeps those whose title or messages contain it. Ones never written in (no title) are
    left out."""
    like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    rows = conn.execute(
        """
        SELECT c.id, c.title, c.agent_id, c.created_at, c.updated_at, c.page, c.page_title
        FROM conversation c
        WHERE c.title != '' AND (
            :query = '' OR c.title LIKE :like ESCAPE '\\' OR EXISTS (
                SELECT 1 FROM message m WHERE m.conversation_id = c.id
                AND m.role IN ('user', 'assistant') AND m.content LIKE :like ESCAPE '\\'))
        ORDER BY c.updated_at DESC, c.id DESC
        LIMIT :limit
        """,
        {"query": query, "like": like, "limit": limit},
    )
    return [dict(row) for row in rows]


def history(conn: sqlite3.Connection, query: str = "") -> dict:
    """The chat history (search chats dialog, /api/agent/conversations): chats and the open one."""
    return {"conversations": conversations(conn, query), "current": current_conversation(conn)}


def conversation_page(conn: sqlite3.Connection, conversation_id: int) -> str | None:
    """The page the conversation was last open on: where to continue it."""
    row = conn.execute("SELECT page FROM conversation WHERE id = ?", (conversation_id,)).fetchone()
    return row["page"] if row else None


def set_page(conn: sqlite3.Connection, conversation_id: int, page: str, page_title: str) -> None:
    """The conversation is open on this page now; written only when it changed."""
    with conn:
        conn.execute(
            "UPDATE conversation SET page = ?, page_title = ? "
            "WHERE id = ? AND (page IS NOT ? OR page_title IS NOT ?)",
            (page, page_title, conversation_id, page, page_title),
        )


def conversation_agent(conn: sqlite3.Connection, conversation_id: int | None) -> str | None:
    """The agent a conversation talks to (None: the default agent)."""
    row = conn.execute("SELECT agent_id FROM conversation WHERE id = ?", (conversation_id,)).fetchone()
    return row["agent_id"] if row else None


def set_conversation_agent(conn: sqlite3.Connection, agent_id: str) -> None:
    """Switch the current conversation (started if there is none) to an agent, from its next message."""
    if not any(a.id == agent_id for a in settings.get().agents):
        raise ValueError("There is no such agent.")
    conversation_id = current_conversation(conn) or new_conversation(conn)
    with conn:
        conn.execute("UPDATE conversation SET agent_id = ? WHERE id = ?", (agent_id, conversation_id))


def _messages(conn: sqlite3.Connection, conversation_id: int | None, tool_results: bool = True) -> list[dict]:
    """The conversation's messages in order, tool calls parsed. tool_results=False leaves out
    the (possibly large) results, which only the model needs."""
    content = "content" if tool_results else "CASE role WHEN 'tool' THEN '' ELSE content END AS content"
    rows = conn.execute(
        f"SELECT role, {content}, tool_calls, tool_call_id, status, error, agent, model FROM message "
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
                result.append({"role": "agent", "parts": [], "status": "done", "error": None, "agent": None, "model": None})
            turn = result[-1]
            if m["content"]:
                turn["parts"].append({"kind": "text", "text": m["content"]})
            for call in m["tool_calls"]:
                turn["parts"].append({"kind": "tool", "name": call["name"], "status": tool_status.get(call["id"], "error")})
            turn["status"], turn["error"] = m["status"], m["error"]
            turn["agent"], turn["model"] = m["agent"] or turn["agent"], m["model"] or turn["model"]
    return result


def _history(conn: sqlite3.Connection, conversation_id: int, agent: settings.Agent) -> list[dict]:
    """The conversation as the model gets it. A tool call without a saved result (the app stopped
    in between) is left out, since the API requires every call to have its result."""
    messages = _messages(conn, conversation_id)
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    history = [{"role": "system", "content": system_prompt(agent)}]
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


async def stream_turn(text: str, page: str | None = None, page_title: str | None = None) -> AsyncIterator[dict]:
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
            conversation_id = current_conversation(conn) or new_conversation(conn)
        turn = run_turn(conversation_id, text, page, page_title)
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


def run_turn(conversation_id: int, text: str, page: str | None = None, page_title: str | None = None) -> Iterator[dict]:
    """Send the person's message (written on `page`, which the conversation keeps as where it
    was left) to the conversation's agent and yield what happens, for the chat to show as it streams:
      {"type": "agent", "agent": "Assistant", "model": "Qwen 3 8B"}   who answers (first)
      {"type": "text", "delta": "..."}                      part of the reply
      {"type": "tool", "id", "name", "status": "running" | "done" | "error"}
      {"type": "done"} or {"type": "error", "message": "..."}
    If the generator is closed early (a stop), the reply so far is saved as stopped.
    """
    with closing(db.connect()) as conn:
        with conn:
            conn.execute("INSERT INTO message (conversation_id, role, content) VALUES (?, 'user', ?)", (conversation_id, text))
            conn.execute(
                f"UPDATE conversation SET updated_at = {db.NOW}, title = CASE title WHEN '' THEN ? ELSE title END, "
                "page = coalesce(?, page), page_title = coalesce(?, page_title) WHERE id = ?",
                (text[:TITLE_LENGTH], page, page_title, conversation_id),
            )

        found = settings.find_agent(conversation_agent(conn, conversation_id))
        who = (found[0].name, found[1].name) if found else (None, None)

        def fail(message: str) -> dict:
            """A turn that ends before any reply: an error message of its own."""
            with conn:
                conn.execute(
                    "INSERT INTO message (conversation_id, role, status, error, agent, model) "
                    "VALUES (?, 'assistant', 'error', ?, ?, ?)",
                    (conversation_id, message, *who),
                )
            return {"type": "error", "message": message}

        if not found:
            yield fail("Set up an agent first: Settings → Models, then Agents.")
            return
        agent, model = found
        yield {"type": "agent", "agent": agent.name, "model": model.name}

        with client(model) as model_client:
            for _ in range(MAX_ROUNDS):
                reply_id = conn.execute(
                    "INSERT INTO message (conversation_id, role, status, agent, model) VALUES (?, 'assistant', 'streaming', ?, ?)",
                    (conversation_id, *who),
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
                        model=model.model, messages=_history(conn, conversation_id, agent), tools=agent_tools.schemas(),
                        stream=True,
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
                    finish("error", describe(exc, model))
                    yield {"type": "error", "message": describe(exc, model)}
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
