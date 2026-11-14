"""Settings pages. Forms post, then redirect back (to where the user came from, or to the same
section with what happened). Models and agents have the same actions as JSON in routes/agents.py."""

from typing import Annotated, get_args
from urllib.parse import urlencode

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from app import agent, agents, config, settings
from app.agents import AgentForm, ModelForm
from app.routes import message, safe_next
from app.templating import templates

router = APIRouter(prefix="/settings")

GENERAL = {"label": "General", "href": "/settings/general"}
MODELS = {"label": "Models", "href": "/settings/models"}
AGENTS = {"label": "Agents", "href": "/settings/agents"}
SECTIONS = [GENERAL, MODELS, AGENTS]


def _back(section: dict, next: str | None, action=None, **query) -> RedirectResponse:
    """Run `action` (a settings change), then go back to the section, with its error if it failed."""
    try:
        if action:
            action()
    except ValueError as exc:
        query["error"] = message(exc)
    return RedirectResponse(f"{section['href']}?{urlencode({'next': safe_next(next)} | query)}", status_code=303)


def _page(request: Request, template: str, section: dict, next: str | None, **context):
    return templates.TemplateResponse(
        request, template, {"sections": SECTIONS, "section": section, "next": safe_next(next), **context}
    )


@router.get("/general", response_class=HTMLResponse)
def general(request: Request, next: str | None = None):
    return _page(request, "settings/general.html", GENERAL, next, current=settings.get(),
                 themes=[(t, t.capitalize()) for t in get_args(settings.Theme)], data_dir=config.DATA_DIR)


@router.post("/general")
def save_general(
    theme: Annotated[settings.Theme, Form()],
    next: Annotated[str | None, Form()] = None,
):
    settings.update(theme=theme)
    return RedirectResponse(safe_next(next), status_code=303)


# --- Models ---


@router.get("/models", response_class=HTMLResponse)
def models_page(request: Request, next: str | None = None, error: str | None = None,
                tested: str | None = None, test: str | None = None, ok: bool = False):
    return _page(request, "settings/models.html", MODELS, next, models=settings.get().models, error=error,
                 test={"model": tested, "message": test, "ok": ok} if tested else None)


@router.post("/models")
def add_model(form: Annotated[ModelForm, Form()], next: Annotated[str | None, Form()] = None):
    return _back(MODELS, next, lambda: agents.add_model(form))


@router.post("/models/{model_id}")
def edit_model(model_id: str, form: Annotated[ModelForm, Form()], next: Annotated[str | None, Form()] = None):
    return _back(MODELS, next, lambda: agents.update_model(model_id, form))


@router.post("/models/{model_id}/delete")
def delete_model(model_id: str, next: Annotated[str | None, Form()] = None):
    return _back(MODELS, next, lambda: agents.delete_model(model_id))


@router.post("/models/{model_id}/test")
def test_model(model_id: str, next: Annotated[str | None, Form()] = None):
    model = settings.find_model(model_id)
    if model is None:
        return _back(MODELS, next, error="There is no such model.")
    ok, result = agent.test_connection(model)
    return _back(MODELS, next, tested=model_id, test=result, ok=ok)


# --- Agents ---


@router.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request, next: str | None = None, error: str | None = None):
    s = settings.get()
    return _page(request, "settings/agents.html", AGENTS, next, agents=s.agents, models=s.models,
                 model_names={m.id: m.name for m in s.models}, default_agent=s.default_agent, error=error)


@router.post("/agents")
def add_agent(form: Annotated[AgentForm, Form()], next: Annotated[str | None, Form()] = None):
    return _back(AGENTS, next, lambda: agents.add_agent(form))


@router.post("/agents/{agent_id}")
def edit_agent(agent_id: str, form: Annotated[AgentForm, Form()], next: Annotated[str | None, Form()] = None):
    return _back(AGENTS, next, lambda: agents.update_agent(agent_id, form))


@router.post("/agents/{agent_id}/delete")
def delete_agent(agent_id: str, next: Annotated[str | None, Form()] = None):
    return _back(AGENTS, next, lambda: agents.delete_agent(agent_id))


@router.post("/agents/{agent_id}/default")
def default_agent(agent_id: str, next: Annotated[str | None, Form()] = None):
    return _back(AGENTS, next, lambda: agents.set_default(agent_id))


UI_FIELDS = {"sidebar_width", "sidebar_collapsed"}


@router.post("/ui", status_code=204)
def save_ui(change: dict = Body()):
    """Saves how the UI was left (the sidebar resized or collapsed), from the page's JS, so the next
    page renders the same way."""
    if not change or not change.keys() <= UI_FIELDS:
        raise HTTPException(422, f"expected only {sorted(UI_FIELDS)}")
    try:
        settings.update(**change)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False))
