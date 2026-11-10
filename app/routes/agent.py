"""The agent chat: send a message (the reply streams back), read the conversation, start a new chat
or open a past one."""

import json
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from app import agent, config, db
from app.routes import safe_next
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


class NewMessage(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
    # The page it was written on: its path and the browser tab's title ("COCO val · babasu").
    page: str = "/"
    page_title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""


def _page_title(title: str) -> str:
    """A page's own name from its tab title: "COCO val · babasu" -> "COCO val"; Home is just "babasu"."""
    name = title.removesuffix(f" · {config.APP_NAME}")
    return "Home" if name in ("", config.APP_NAME) else name


@router.post("/api/agent/messages")
async def send_message(message: NewMessage) -> StreamingResponse:
    """Streams the turn as newline-delimited JSON events (see agent.run_turn). Dropping the
    request (the stop button) stops the turn."""
    turn = agent.stream_turn(message.text, safe_next(message.page), _page_title(message.page_title))
    events = (json.dumps(event) + "\n" async for event in turn)
    return StreamingResponse(events, media_type="application/x-ndjson")


@router.get("/api/agent/messages")
def conversation(conn: DB) -> list[dict]:
    """The current conversation, as the chat shows it."""
    return agent.turns(conn, agent.current_conversation(conn))


@router.get("/api/agent/conversations")
def conversations(conn: DB, q: str = "") -> dict:
    """Past conversations, most recent first, each with the page it was last written on, and the
    one open in the chat (null: a new chat). `q` searches titles and messages."""
    return agent.history(conn, q.strip())


@router.get("/agent/history", response_class=HTMLResponse)
def history(request: Request, conn: DB, q: str = ""):
    """The search chats dialog's results (partials/chat-history.html)."""
    return templates.TemplateResponse(request, "partials/chat-history.html", agent.history(conn, q.strip()) | {"query": q.strip()})


class OpenConversation(BaseModel):
    conversation_id: int | None  # null: a new chat


@router.put("/api/agent/current", status_code=204)
def open_conversation(change: OpenConversation, conn: DB) -> None:
    """Open a past conversation in the chat, or start a new one."""
    try:
        agent.open_conversation(conn, change.conversation_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


class ConversationAgent(BaseModel):
    agent_id: str


@router.put("/api/agent/conversation", status_code=204)
def switch_agent(change: ConversationAgent, conn: DB) -> None:
    """Switch the current chat to another agent; it answers from the next message on."""
    try:
        agent.set_conversation_agent(conn, change.agent_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/agent/open")
def open_chat(conn: DB, conversation_id: Annotated[int | None, Form()] = None, next: Annotated[str, Form()] = "/"):
    """New chat (no id: stays on the page), or a chat picked from the search chats dialog: it opens
    on the page it was last written on. One that's gone (deleted meanwhile) is simply not opened."""
    try:
        agent.open_conversation(conn, conversation_id)
    except ValueError:
        return RedirectResponse(safe_next(next), status_code=303)
    page = agent.conversation_page(conn, conversation_id) if conversation_id is not None else None
    return RedirectResponse(safe_next(page or next), status_code=303)
