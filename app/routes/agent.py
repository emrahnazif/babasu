"""The agent chat: send a message (the reply streams back), start a new chat, read the conversation."""

import json
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Form
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
    return agent.turns(conn, agent.latest_conversation(conn))


@router.post("/agent/new")
def new_chat(conn: DB, next: Annotated[str, Form()] = "/"):
    agent.new_conversation(conn)
    return RedirectResponse(safe_next(next), status_code=303)
