"""The agent chat: send a message (the reply streams back), read the conversation, start a new chat
or open a past one."""

import json
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from app import agent, db
from app.routes import safe_next

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


class NewMessage(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


@router.post("/api/agent/messages")
async def send_message(message: NewMessage) -> StreamingResponse:
    """Streams the turn as newline-delimited JSON events (see agent.run_turn). Dropping the
    request (the stop button) stops the turn."""
    events = (json.dumps(event) + "\n" async for event in agent.stream_turn(message.text))
    return StreamingResponse(events, media_type="application/x-ndjson")


@router.get("/api/agent/messages")
def conversation(conn: DB) -> list[dict]:
    """The current conversation, as the chat shows it."""
    return agent.turns(conn, agent.current_conversation(conn))


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
    """New chat (no id), or a past chat. One that's gone (deleted meanwhile) is simply not opened."""
    try:
        agent.open_conversation(conn, conversation_id)
    except ValueError:
        pass
    return RedirectResponse(safe_next(next), status_code=303)
