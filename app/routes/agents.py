"""Models and agents as JSON, for the agent and other clients; the same actions as Settings →
Models and Agents. API keys are never returned, only whether one is saved."""

from fastapi import APIRouter, HTTPException

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
    return [_model(m) for m in settings.get().models]


@router.post("/models", status_code=201)
def add_model(form: ModelForm) -> dict:
    return _model(_do(lambda: agents.add_model(form)))


@router.put("/models/{model_id}", status_code=204)
def edit_model(model_id: str, form: ModelForm) -> None:
    _do(lambda: agents.update_model(model_id, form))


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str) -> None:
    _do(lambda: agents.delete_model(model_id))


@router.post("/models/{model_id}/test")
def test_model(model_id: str) -> dict:
    model = settings.find_model(model_id)
    if model is None:
        raise HTTPException(404, "There is no such model.")
    ok, result = agent.test_connection(model)
    return {"ok": ok, "message": result}


@router.get("/agents")
def list_agents() -> dict:
    s = settings.get()
    return {"agents": [a.model_dump() for a in s.agents], "default": s.default_agent}


@router.post("/agents", status_code=201)
def add_agent(form: AgentForm) -> dict:
    return _do(lambda: agents.add_agent(form)).model_dump()


@router.put("/agents/{agent_id}", status_code=204)
def edit_agent(agent_id: str, form: AgentForm) -> None:
    _do(lambda: agents.update_agent(agent_id, form))


@router.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: str) -> None:
    _do(lambda: agents.delete_agent(agent_id))


@router.post("/agents/{agent_id}/default", status_code=204)
def default_agent(agent_id: str) -> None:
    _do(lambda: agents.set_default(agent_id))
