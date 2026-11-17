"""The agent: a conversation with a model through an OpenAI-compatible chat completions API.

The model can call babasu's own functions as tools (app/agent_tools.py). One turn: save the
person's message, ask the model, stream its text, run the tools it asks for, ask again, until it
answers without tools. Every step is saved as it happens, so the chat shows what happened even after
an error or a stop.

A turn belongs to the app, not to the page that sent the message: it runs in its own thread and
keeps the events it has sent, so any page can follow it (from its first event, then live). Leaving
a page stops nothing; stop is asked for (stop_turn), or the app closing stops every turn.
"""

import asyncio
import base64
import json
import queue
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import closing

import openai

from app import agent_tools, code, config, db, settings

MAX_ROUNDS = 20  # model calls per turn; each round may run tools (work of several steps takes many)
TITLE_LENGTH = 60

# What every agent is told about the app; the agent's own instructions follow.
APP_PROMPT = f"""You are the person's assistant, working with them inside {config.APP_NAME}: a local computer vision \
app where people import images into projects, annotate them (by hand, or with vision models) and run workflows; \
training models comes later.

You're the same capable assistant they'd talk to anywhere else: answer any question fully, at the length it needs \
(a how-to gets its steps and examples), and write and explain code when that's what they ask for. What's more here \
is that you can work in their app: your tools do everything its pages do, and more (code over many images, files). \
The person doesn't see or use your tools: you do the work. So talk with them about what's in their app (projects, \
images, classes, runs), never with steps for them to click or code to paste, and with no tool names or API paths, \
unless they ask you for code.

How you work here:
- Their data first: before you advise on it or act on it, look at it. Each message starts with the page it was \
written on and what's on it; that page's data is at the same path under /api (/projects/3/images/42 → \
/api/projects/3/images/42); "this image", "this project", "these" are that page's. find_api finds any other \
operation of the API and what it takes.
- They watch you work: each step shows as one line (what it does, what came out, the images it found). When why \
you take a step isn't obvious, say it in one short sentence first.
- Answer briefly: the result is in the app, so link to it ([the 312 night images](/projects/3?where=night)) \
instead of listing it.
- When they ask how to do something you can do here, look first, then offer to do it; when they ask for it, do it, \
and say in a sentence what you did.
- "Change", "add", "rename", "remove": do it (find_api has what an operation takes); the page they're on follows.
- A name: look it up (search, find_images) first. Ask only when it still means more than one thing, saying what \
you found; don't ask what has a sensible default.
- Changes to projects, images, annotations and runs, and deleted workflows, can be undone for 30 days (History): \
delete when asked, and say what went.
- A job that downloads models first (downloads_first in what starts it): say which, and how big.
- What you can't do: say so, and where in the app it's done.
- Work of several steps (reading many images or results, comparing, measuring, then deciding): work like a \
programmer in this chat's folder (read_docs code). Your scripts read through the API and save what they find in \
files; you read what they print, decide, and go on. Data moves through code and files, never through your words: \
don't copy ids or numbers out of what was printed. Check each count printed against what you expected."""


class Busy(Exception):
    """A message sent while the conversation's agent is still answering the last one."""


# Events that follow one of their kind (of the same tool call, for a call's) are merged into it:
# text joined (by this key); progress replaced (only the latest counts).
_MERGED = {"text": "delta", "call": "delta", "output": "text", "progress": None}


class Turn:
    """One turn in progress (or just ended): the events it has sent, who follows it, its stop flag."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.ended = False
        self.stop = threading.Event()
        self.followers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

    def send(self, event: dict | None) -> None:
        """Keep an event and pass it on to every follower; None: the turn has ended."""
        with _lock:
            if event is None:
                self.ended = True
            elif (event["type"] in _MERGED and self.events and self.events[-1]["type"] == event["type"]
                  and self.events[-1].get("id") == event.get("id")):
                # Kept as one event, so a page that follows later replays it at once.
                key = _MERGED[event["type"]]
                self.events[-1] = event | ({key: self.events[-1][key] + event[key]} if key else {})
            else:
                self.events.append(event)
            for loop, queue in self.followers:
                loop.call_soon_threadsafe(queue.put_nowait, event)


# Each conversation's latest turn since the app started: running, or ended (kept so a page that
# rendered it running can still follow it to its end). Ended ones go when the next turn starts.
_turns: dict[int, Turn] = {}
_lock = threading.Lock()


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
    with _lock:
        for turn in _turns.values():
            turn.stop.set()


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


def _messages(conn: sqlite3.Connection, conversation_id: int | None, results: bool = False) -> list[dict]:
    """The conversation's messages in order, tool calls parsed. Tool results and the pictures tools
    gave can be large and only the model reads them, and only this turn's (`results`); the others
    are left out."""
    rows = conn.execute(
        """WITH turn AS (SELECT max(id) AS started FROM message WHERE conversation_id = :id AND role = 'user')
           SELECT id, role, tool_calls, tool_call_id, status, error, label, note, images, agent, model, page, page_title,
                  about, tokens_in, tokens_out, length(picture) IS NOT NULL AS has_picture,  -- length() doesn't read it
                  CASE WHEN role = 'tool' AND NOT (:results AND id > turn.started) THEN '' ELSE content END AS content,
                  CASE WHEN :results AND id > turn.started THEN picture END AS picture
           FROM message, turn WHERE conversation_id = :id ORDER BY id""",
        {"id": conversation_id, "results": results},
    ).fetchall()
    return [dict(row) | {"tool_calls": json.loads(row["tool_calls"] or "[]")} for row in rows]


def turns(conn: sqlite3.Connection, conversation_id: int | None) -> list[dict]:
    """The conversation as the chat shows it: the person's turns, and each agent turn as one
    entry with its parts in order (its text, and the tool calls that ran: each its label, note,
    status, the images it found, {"project_id", "media_ids"}, and the picture it gave the model, its
    address), how it ended, and the tokens its model read and wrote in all."""
    messages = _messages(conn, conversation_id)
    answers = {m["tool_call_id"]: m for m in messages if m["role"] == "tool"}
    result: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            result.append({"role": "user", "text": m["content"]})
        elif m["role"] == "assistant":
            if not result or result[-1]["role"] != "agent":
                result.append({"role": "agent", "parts": [], "status": "done", "error": None, "agent": None, "model": None,
                               "tokens_in": 0, "tokens_out": 0})
            turn = result[-1]
            if m["content"]:
                turn["parts"].append({"kind": "text", "text": m["content"]})
            for answer in filter(None, (answers.get(call["id"]) for call in m["tool_calls"])):
                turn["parts"].append({"kind": "tool", "label": answer["label"] or "", "note": answer["note"] or "",
                                      "status": answer["status"],
                                      "images": json.loads(answer["images"]) if answer["images"] else None,
                                      "picture": picture_url(answer["id"]) if answer["has_picture"] else None})
            turn["status"], turn["error"] = m["status"], m["error"]
            turn["agent"], turn["model"] = m["agent"] or turn["agent"], m["model"] or turn["model"]
            turn["tokens_in"] += m["tokens_in"] or 0
            turn["tokens_out"] += m["tokens_out"] or 0
    return result


def picture_url(message_id: int) -> str:
    """Where the picture a tool call gave the model is shown (routes/agent.py)."""
    return f"/agent/pictures/{message_id}"


def _seen(pictures: list[tuple[str, bytes]]) -> dict:
    """The pictures this turn's tool calls gave the model, as a message of the person's: the API
    takes images there, not in a tool's result."""
    parts = [{"type": "text", "text": "(Not from the person: the pictures you asked to look at, in order: "
              + "; ".join(what for what, _ in pictures) + ". Go on with what they asked.)"}]
    parts += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}
              for _, jpeg in pictures]
    return {"role": "user", "content": parts}


def _on_page(message: dict, current: bool) -> str:
    """The line the model gets before a message of the person's: the page it was written on, and
    for the current one what that page showed (its own words, templates' `about` block)."""
    if not message["page"]:
        return ""
    if current and message["about"]:
        return f"(On the page {message['page']}: {message['about']})\n\n"
    title = f' "{message["page_title"]}"' if message["page_title"] else ""
    return f"(On the page{title}, {message['page']}.)\n\n"


def _arguments(text: str) -> str:
    """A tool call's arguments as the API takes them back: a JSON object. Ones that aren't (cut
    off, malformed, empty) go as {}, or the server refuses the whole conversation; the call's
    result already told the model what was wrong."""
    try:
        return text if isinstance(json.loads(text), dict) else "{}"
    except ValueError:
        return "{}"


LONG = 200  # characters of an argument an earlier call keeps; a longer one (code, a file) is left out
PICTURES = 4  # pictures of this turn the model gets at most, the latest; it read the others in their round


def _shortened(text: str) -> str:
    """An earlier call's arguments: what it was called on (ids, filters, paths) kept, its long
    texts (code, a file's content) left out."""
    return json.dumps({key: value if len(json.dumps(value)) <= LONG else "(left out)"
                       for key, value in json.loads(_arguments(text)).items()}, ensure_ascii=False)


def _history(conn: sqlite3.Connection, conversation_id: int, agent: settings.Agent) -> list[dict]:
    """The conversation as the model gets it. A tool call without a saved result (the app stopped
    in between) is left out, since the API requires every call to have its result.

    The model works with a whole result only in the turn it comes in. An earlier turn's calls keep
    what they were called on and what came out (their note): what the model concluded is in its
    reply, data that has to last is in its folder's files, and it can call again. So a long chat
    doesn't push its start (the instructions) out of a small context. Pictures tools gave go only
    with this turn, its latest PICTURES."""
    messages = _messages(conn, conversation_id, results=True)
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    this_turn = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=0)
    latest = {m["id"] for m in [m for m in messages if m["picture"]][-PICTURES:]}
    history = [{"role": "system", "content": system_prompt(agent)}]
    pictures: list[tuple[str, bytes]] = []  # sent after the results of the calls that gave them
    for i, m in enumerate(messages):
        earlier = i < this_turn
        if m["role"] == "user":
            history.append({"role": "user", "content": _on_page(m, current=i == this_turn) + m["content"]})
        elif m["role"] == "assistant":
            calls = [c for c in m["tool_calls"] if c["id"] in answered]
            if not m["content"] and not calls:
                continue
            message = {"role": "assistant", "content": m["content"]}
            if calls:
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {
                        "name": c["name"], "arguments": (_shortened if earlier else _arguments)(c["arguments"])}}
                    for c in calls
                ]
            history.append(message)
        else:
            content = f"(An earlier result, shortened) {m['note'] or m['status']}" if earlier else m["content"]
            history.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": content})
            if m["id"] in latest:
                pictures.append((m["note"], m["picture"]))
            if pictures and (i + 1 == len(messages) or messages[i + 1]["role"] != "tool"):
                history.append(_seen(pictures))  # after the last result of these calls
                pictures = []
    return history


# --- One turn ---


def _streamed(open_stream: Callable, stop: threading.Event) -> Iterator:
    """The model's reply, chunk by chunk, read in a thread of its own so that a stop is heard at
    once, even while the server is still reading the conversation (before its first chunk, which
    takes long on a slow local model). Once the turn stops reading (a stop, or anything else that
    ends it), the reader closes the stream at its next chunk."""
    chunks: queue.Queue = queue.Queue()
    left = threading.Event()  # the turn stopped reading

    def read() -> None:
        try:
            with open_stream() as stream:
                for chunk in stream:
                    if left.is_set():
                        return
                    chunks.put(chunk)
        except Exception as exc:
            chunks.put(exc)
        finally:
            chunks.put(None)

    threading.Thread(target=read, name="agent-reply", daemon=True).start()
    try:
        while not stop.is_set():
            try:
                chunk = chunks.get(timeout=code.STOP_CHECK)
            except queue.Empty:
                continue
            if chunk is None:
                return
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
    finally:
        left.set()


def start_turn(text: str, page: str | None = None, page_title: str | None = None, about: str | None = None) -> Turn:
    """Send the person's message to the current conversation (started if there is none) and run
    the turn in its own thread (see run_turn). Busy if that conversation's last turn is still running."""
    with closing(db.connect()) as conn:
        conversation_id = current_conversation(conn) or new_conversation(conn)
    turn = Turn()
    with _lock:
        running = _turns.get(conversation_id)
        if running and not running.ended:
            raise Busy("The agent is still answering. Wait for it, or stop it first.")
        for done in [id for id, t in _turns.items() if t.ended]:
            del _turns[done]
        _turns[conversation_id] = turn

    def run() -> None:
        try:
            for event in run_turn(conversation_id, text, turn.stop, page, page_title, about):
                turn.send(event)
        finally:
            turn.send(None)

    threading.Thread(target=run, name="agent-turn", daemon=True).start()
    return turn


def latest_turn(conversation_id: int | None) -> Turn | None:
    """The conversation's latest turn since the app started, running or just ended."""
    with _lock:
        return _turns.get(conversation_id)


def running(conversation_id: int | None) -> bool:
    """Whether the conversation's agent is answering now."""
    turn = latest_turn(conversation_id)
    return turn is not None and not turn.ended


def stop_turn(conversation_id: int | None) -> None:
    """Ask the conversation's running turn to stop; it saves what it has as stopped at its next step."""
    if turn := latest_turn(conversation_id):
        turn.stop.set()


async def follow(turn: Turn) -> AsyncIterator[dict]:
    """The turn's events from the first, then each as it comes, until it ends. A follower going
    away (a page change) leaves the turn running."""
    queue: asyncio.Queue = asyncio.Queue()
    with _lock:
        past, ended = list(turn.events), turn.ended
        follower = (asyncio.get_running_loop(), queue)
        if not ended:
            turn.followers.append(follower)
    try:
        for event in past:
            yield event
        if not ended:
            while (event := await queue.get()) is not None:
                yield event
    finally:
        with _lock:
            if follower in turn.followers:
                turn.followers.remove(follower)


def run_turn(
    conversation_id: int, text: str, stop: threading.Event, page: str | None = None, page_title: str | None = None,
    about: str | None = None,
) -> Iterator[dict]:
    """Send the person's message (written on `page`, which the conversation keeps as where it
    was left; `about`: what that page showed, in its own words) to the conversation's agent and yield what happens, for the chat to show as it streams:
      {"type": "agent", "agent": "Assistant", "model": "Qwen 3 8B"}   who answers (first)
      {"type": "text", "delta": "..."}                      part of the reply
      {"type": "tool", "id", "label", "status": "running" | "done" | "error", "note", "images", "picture"}
      {"type": "call", "id", "name", "delta": "..."}        a tool call's arguments as the model writes them
      {"type": "output", "text": "..."}                     what the running tool prints (run_code)
      {"type": "progress", "done": 3, "total": 10}          how far the running tool is
      {"type": "usage", "tokens_in", "tokens_out"}          what a call of the model read and wrote
      {"type": "open", "url": "/projects/3?where=car"}      a page to show once the reply is done
      {"type": "changed"}                                   a tool changed what pages show: once the
                                                            reply is done, reload (unless one opens)
      {"type": "done"}, {"type": "stopped"} or {"type": "error", "message": "..."}
    Once `stop` is set, the turn ends at once (a tool running ends too) and saves the reply so far
    as stopped. What the turn changes is one action in the history, named by the message.
    """
    with closing(db.connect()) as conn, db.action("agent", text), agent_tools.turn_of(conversation_id, stop):
        with conn:
            conn.execute(
                "INSERT INTO message (conversation_id, role, content, page, page_title, about) VALUES (?, 'user', ?, ?, ?, ?)",
                (conversation_id, text, page, page_title, about),
            )
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
                ended = None  # why the model stopped writing: "stop", "tool_calls", "length"…
                usage = None  # what the server counted, in the stream's last chunk

                def finish(status: str, error: str | None = None, tool_calls: list | None = None) -> None:
                    with conn:
                        conn.execute(
                            "UPDATE message SET content = ?, tool_calls = ?, status = ?, error = ?, tokens_in = ?, "
                            "tokens_out = ? WHERE id = ?",
                            ("".join(text_parts), json.dumps(tool_calls) if tool_calls else None, status, error,
                             usage and usage.prompt_tokens, usage and usage.completion_tokens, reply_id),
                        )

                def stopped(ran: list | None = None) -> dict:
                    """Stopped: keep the text that arrived and the tool calls that ran."""
                    finish("stopped", tool_calls=ran)
                    return {"type": "stopped"}

                messages = _history(conn, conversation_id, agent)
                try:
                    for chunk in _streamed(lambda: model_client.chat.completions.create(
                            model=model.model, messages=messages, tools=agent_tools.schemas(model.sees_images), stream=True,
                            stream_options={"include_usage": True}), stop):
                        usage = chunk.usage or usage
                        if not chunk.choices:
                            continue
                        ended = chunk.choices[0].finish_reason or ended
                        delta = chunk.choices[0].delta
                        if delta.content:
                            text_parts.append(delta.content)
                            yield {"type": "text", "delta": delta.content}
                        for part in delta.tool_calls or []:
                            # Its id is ours, from the reply's: servers may send none, or the same each round.
                            call = calls.setdefault(part.index, {"id": f"call_{reply_id}_{part.index}", "name": "", "arguments": ""})
                            if part.function:
                                call["name"] += part.function.name or ""
                                call["arguments"] += part.function.arguments or ""
                                if part.function.arguments:  # a page may show it as it's written
                                    yield {"type": "call", "id": call["id"], "name": call["name"],
                                           "delta": part.function.arguments}
                except Exception as exc:  # unreachable server, bad key, unknown model…
                    finish("error", describe(exc, model))
                    yield {"type": "error", "message": describe(exc, model)}
                    return
                if usage:
                    yield {"type": "usage", "tokens_in": usage.prompt_tokens, "tokens_out": usage.completion_tokens}
                if stop.is_set():
                    yield stopped()
                    return

                if ended == "length":  # cut off mid-reply: a tool call in it may be half written
                    message = f"{model.name} reached its output limit mid-reply. Ask it to go on, or for less at once."
                    finish("error", message)
                    yield {"type": "error", "message": message}
                    return
                tool_calls = list(calls.values())
                if not tool_calls and not text_parts:  # e.g. a model that put its tool call in its thinking and stopped
                    message = f"{model.name} answered nothing (no text, no tool call). Send it again, or try another model."
                    finish("error", message)
                    yield {"type": "error", "message": message}
                    return
                finish("done", tool_calls=tool_calls)
                if not tool_calls:
                    yield {"type": "done"}
                    return

                for ran, call in enumerate(tool_calls):
                    label = agent_tools.label(call["name"], call["arguments"])
                    yield {"type": "tool", "id": call["id"], "label": label, "status": "running"}
                    try:
                        # What it shows while it works (run_code's output), then its result.
                        outcome, status, error = (yield from agent_tools.call(conn, call["name"], call["arguments"])), "done", None
                    except Exception as exc:  # the model sees the error and can correct itself
                        error = str(exc)
                        outcome, status = agent_tools.Outcome(json.dumps({"error": error}), error.partition("\n")[0]), "error"
                    with conn:
                        answer_id = conn.execute(
                            "INSERT INTO message (conversation_id, role, content, tool_call_id, status, error, label, note, "
                            "images, picture) VALUES (?, 'tool', ?, ?, ?, ?, ?, ?, ?, ?)",
                            (conversation_id, outcome.result, call["id"], status, error, label, outcome.note,
                             json.dumps(outcome.images) if outcome.images else None, outcome.picture),
                        ).lastrowid
                    yield {"type": "tool", "id": call["id"], "label": label, "status": status, "note": outcome.note,
                           "images": outcome.images, "picture": picture_url(answer_id) if outcome.picture else None}
                    if stop.is_set():  # the tools after it don't run, and the model isn't asked again
                        yield stopped(tool_calls[:ran + 1])
                        return
                    if status != "done":
                        continue
                    found = agent_tools.TOOLS[call["name"]]
                    if found.opens and "open" in (opened := json.loads(outcome.result)):
                        yield {"type": "open", "url": opened["open"]}
                    elif agent_tools.writes(call["name"], call["arguments"]):
                        yield {"type": "changed"}

        yield fail(f"Stopped after {MAX_ROUNDS} rounds of tool calls without an answer.")
