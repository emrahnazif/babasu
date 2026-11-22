"""Setting up agents and the models they run on: add, change, delete, test (Settings → Models and
Agents, and the same as JSON). They're stored in settings.json (app/settings.py); running an agent
is app/agent.py. Every change is checked by the Settings model; problems raise ValueError with a
message for people.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app import agent, settings
from app.settings import Agent, Blank, Model, Settings, Thinking, slug


class ModelForm(BaseModel):
    """A language model: a connection to an OpenAI-compatible chat completions API."""

    name: str = Field(description="Its name for people, e.g. Qwen 3 8B.")
    base_url: str = Field(description="The API's address, e.g. http://localhost:11434/v1 (Ollama).")
    model: str = Field(description="The server's name for the model, e.g. qwen3:8b.")
    api_key: str = Field("", description="The API key, if the server needs one; blank on a change keeps the saved one.")
    sees_images: bool = Field(True, description="It reads images: the agent can look at images and crops (they're "
                              "sent to its server).")
    thinking: Annotated[Thinking | None, Blank] = Field(  # the form's "As the server does": none sent
        None, description="How much it thinks before it replies: none, low, medium or high (sent as "
                          "reasoning_effort); null leaves it to the server.")


class AgentForm(BaseModel):
    """An agent: who answers in the chat."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Its name, e.g. Assistant.")
    model_id: str = Field(description="The model it runs on, an id from GET /api/models.")
    instructions: str = Field("", description="Its role and rules, e.g. Answer from the app's real data.")


def shown_model(m: Model) -> dict:
    """A model as the API and Settings show it: never its API key, only whether one is saved."""
    return m.model_dump(exclude={"api_key"}) | {"has_api_key": bool(m.api_key)}


def shown_agent(a: Agent, default_agent: str) -> dict:
    """An agent as the API and Settings show it, with whether new chats start with it."""
    return a.model_dump() | {"default": a.id == default_agent}


def _find(items: list, item_id: str, kind: str):
    for item in items:
        if item.id == item_id:
            return item
    raise ValueError(f"There is no such {kind}.")


def _save(s: Settings) -> None:
    settings.save(Settings.model_validate(s.model_dump()))  # re-checks ids and references


def add_model(form: ModelForm) -> Model:
    s = settings.get()
    model = Model(id=slug(form.name, {m.id for m in s.models}), **form.model_dump())
    s.models.append(model)
    _save(s)
    return model


def update_model(model_id: str, form: ModelForm) -> None:
    s = settings.get()
    model = _find(s.models, model_id, "model")
    changes = form.model_dump(exclude={"api_key"} if not form.api_key else set())
    s.models[s.models.index(model)] = Model.model_validate(model.model_dump() | changes)
    _save(s)


def delete_model(model_id: str) -> None:
    s = settings.get()
    model = _find(s.models, model_id, "model")
    if users := [a.name for a in s.agents if a.model_id == model_id]:
        raise ValueError(f"This model is used by {', '.join(users)}. Give them another model first.")
    s.models.remove(model)
    _save(s)


def test(form: ModelForm, model_id: str = "") -> tuple[bool, str]:
    """Test a model as typed in the form, before it's saved. Editing a saved model (`model_id`),
    a blank key means its saved key. Raises ValueError if the form isn't a valid model."""
    saved = settings.find_model(model_id) if model_id else None
    fields = form.model_dump() | ({"api_key": saved.api_key} if saved and not form.api_key else {})
    return agent.test_connection(Model(id=model_id or "draft", **fields))


def add_agent(form: AgentForm) -> Agent:
    s = settings.get()
    new = Agent(id=slug(form.name, {a.id for a in s.agents}), **form.model_dump())
    s.agents.append(new)
    _save(s)
    return new


def update_agent(agent_id: str, form: AgentForm) -> None:
    s = settings.get()
    current = _find(s.agents, agent_id, "agent")
    s.agents[s.agents.index(current)] = Agent.model_validate(current.model_dump() | form.model_dump())
    _save(s)


def delete_agent(agent_id: str) -> None:
    s = settings.get()
    current = _find(s.agents, agent_id, "agent")
    if agent_id == s.default_agent and len(s.agents) > 1:
        raise ValueError("This is the default agent. Make another agent the default first.")
    s.agents.remove(current)
    _save(s)


def set_default(agent_id: str) -> None:
    s = settings.get()
    _find(s.agents, agent_id, "agent")
    s.default_agent = agent_id
    _save(s)
