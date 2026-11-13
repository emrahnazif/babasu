"""The agent chat: send a message (the reply streams back), follow or stop the reply in progress,
read the conversation, start a new chat or open a past one."""

import json
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, StringConstraints

from app import agent, db
from app.routes import safe_next
from app.templating import page_name, templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


class NewMessage(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
    # The page it was written on (path and browser tab title), for a chat that starts with this
    # message: page loads record it from then on (templating.agent_chat).
    page: str = "/"
    page_title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""


def _stream(turn: agent.Turn) -> StreamingResponse:
    """The turn's events as newline-delimited JSON (see agent.run_turn), from its first."""
    events = (json.dumps(event) + "\n" async for event in agent.follow(turn))
    return StreamingResponse(events, media_type="application/x-ndjson")


@router.post("/api/agent/messages")
def send_message(message: NewMessage) -> StreamingResponse:
    """Send a message to the current conversation and stream the reply. The reply doesn't depend
    on this request: dropping it (a page change) leaves the agent answering; follow it again with
    GET /api/agent/turn. Refused (409) while the agent is still answering."""
    try:
        turn = agent.start_turn(message.text, safe_next(message.page), page_name(message.page_title))
    except agent.Busy as exc:
        raise HTTPException(409, str(exc)) from None
    return _stream(turn)


@router.get("/api/agent/turn", response_model=None)
def follow_turn(conn: DB) -> StreamingResponse | Response:
    """Follow the current conversation's latest turn: its events from the first, then live until it
    ends (one that already ended: all of them). 204 if it has had none since the app started."""
    turn = agent.latest_turn(agent.current_conversation(conn))
    return _stream(turn) if turn else Response(status_code=204)


@router.post("/api/agent/stop", status_code=204)
def stop_turn(conn: DB) -> None:
    """Stop the current conversation's reply in progress; it ends as stopped at its next step."""
    agent.stop_turn(agent.current_conversation(conn))


@router.get("/api/agent/messages")
def conversation(conn: DB) -> list[dict]:
    """The current conversation, as the chat shows it."""
    return agent.turns(conn, agent.current_conversation(conn))


@router.get("/api/agent/conversations")
def conversations(conn: DB, q: str = "") -> dict:
    """Past conversations, most recent first, each with the page it was last open on, and the
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
    on the page it was last open on. One that's gone (deleted meanwhile) is simply not opened."""
    try:
        agent.open_conversation(conn, conversation_id)
    except ValueError:
        return RedirectResponse(safe_next(next), status_code=303)
    page = agent.conversation_page(conn, conversation_id) if conversation_id is not None else None
    return RedirectResponse(safe_next(page or next), status_code=303)
