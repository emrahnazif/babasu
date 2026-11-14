"""History: what was done in the last 30 days (app/history.py), as a page and as the API the UI and
the agent share, and undoing it."""

import sqlite3
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import db, history
from app.templating import templates

router = APIRouter()
DB = Annotated[sqlite3.Connection, Depends(db.get)]


@router.get("/history", response_class=HTMLResponse)
def history_page(request: Request, conn: DB, error: str | None = None):
    # "done", not "actions": base.html has an `actions` block.
    return templates.TemplateResponse(request, "history.html", {
        "done": history.actions(conn), "days": history.DAYS, "error": error})


@router.post("/history/{action_id}/undo")
def undo_page(action_id: str, conn: DB):
    """Undo from the page, and go back to it (saying why, if it can't be)."""
    try:
        history.undo(conn, action_id)
    except (LookupError, history.Refused) as e:
        return RedirectResponse(f"/history?{urlencode({'error': str(e)})}", status_code=303)
    return RedirectResponse("/history", status_code=303)


@router.get("/api/history")
def api_history(conn: DB, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[dict]:
    """What was done in the last 30 days, newest first: each action (one request, job or agent turn,
    named by its source and label: the request, the job, the person's message to the agent), its
    kind (do, undo, redo), when, what it changed in words, the undo that undid it (`undone_by`) or
    the action an undo undid (`undoes`), and `changed_since` (a later action changed the same
    things, so that one is undone first). Undo one: POST /api/history/{id}/undo."""
    return history.actions(conn, limit)


@router.post("/api/history/{action_id}/undo")
def api_undo(action_id: str, conn: DB) -> dict:
    """Undoes an action: the rows it wrote go back as they were, the files it deleted come back.
    The undo is an action too (undoing it does the action again). Returns it as /api/history shows
    it. 404 if there's no such action; 409 with why if a later action changed the same things
    (undo that first) or something can't go back."""
    try:
        return history.undo(conn, action_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from None
    except history.Refused as e:
        raise HTTPException(409, str(e)) from None
