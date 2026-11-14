"""Models and agents as JSON, for the agent and other clients; the same actions as Settings →
Models and Agents. API keys are never returned, only whether one is saved."""

from fastapi import APIRouter, HTTPException
from pydantic import Field

from app import agent, agents, settings
from app.agents import AgentForm, ModelForm
from app.routes import message

router = APIRouter(prefix="/api")


def _model(m: settings.Model) -> dict:
    return m.model_dump(exclude={"api_key"}) | {"has_api_key": bool(m.api_key)}


def _do(action):
    """Run a settings change; its ValueError becomes a 409 with the message for people."""
    try:
        return action()
    except ValueError as exc:
        raise HTTPException(409, message(exc)) from None


@router.get("/models")
def list_models() -> list[dict]:
    """The language models agents run on: id, name, base_url, model, and whether an API key is saved."""
    return [_model(m) for m in settings.get().models]


@router.post("/models", status_code=201)
def add_model(form: ModelForm) -> dict:
    """Adds a language model; returns it with its id. 409 with why (e.g. a name taken)."""
    return _model(_do(lambda: agents.add_model(form)))


@router.put("/models/{model_id}", status_code=204)
def edit_model(model_id: str, form: ModelForm) -> None:
    """Changes a language model (all its fields; a blank api_key keeps the saved one)."""
    _do(lambda: agents.update_model(model_id, form))


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str) -> None:
    """Deletes a language model. 409 while an agent uses it. Not undone by History (settings)."""
    _do(lambda: agents.delete_model(model_id))


class ModelDraft(ModelForm):
    """A model to test before it's saved."""

    model_id: str = Field("", description="When changing a saved model: its id, so a blank api_key means its saved key.")


@router.post("/models/test")
def test_draft(draft: ModelDraft) -> dict:
    """Test a model before saving it (the Test button in the Add and Edit dialogs)."""
    try:
        ok, result = agents.test(draft, draft.model_id)
    except ValueError as exc:
        ok, result = False, message(exc)
    return {"ok": ok, "message": result}


@router.post("/models/{model_id}/test")
def test_model(model_id: str) -> dict:
    """Checks a saved model answers: {ok, message}."""
    model = settings.find_model(model_id)
    if model is None:
        raise HTTPException(404, "There is no such model.")
    ok, result = agent.test_connection(model)
    return {"ok": ok, "message": result}


@router.get("/agents")
def list_agents() -> dict:
    """The agents (id, name, model_id, instructions) and which is the default for new chats."""
    s = settings.get()
    return {"agents": [a.model_dump() for a in s.agents], "default": s.default_agent}


@router.post("/agents", status_code=201)
def add_agent(form: AgentForm) -> dict:
    """Adds an agent; returns it with its id. 409 with why (e.g. an unknown model)."""
    return _do(lambda: agents.add_agent(form)).model_dump()


@router.put("/agents/{agent_id}", status_code=204)
def edit_agent(agent_id: str, form: AgentForm) -> None:
    """Changes an agent (all its fields)."""
    _do(lambda: agents.update_agent(agent_id, form))


@router.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: str) -> None:
    """Deletes an agent; its chats go on with the default one. 409 for the default agent. Not undone by
    History (settings)."""
    _do(lambda: agents.delete_agent(agent_id))


@router.post("/agents/{agent_id}/default", status_code=204)
def default_agent(agent_id: str) -> None:
    """Makes an agent the default for new chats."""
    _do(lambda: agents.set_default(agent_id))
