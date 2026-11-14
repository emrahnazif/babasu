"""App settings, stored as JSON in the workspace: appearance, and the agents with their models
(how they're stored and checked; adding, changing and deleting them is in app/agents.py).

A model is a connection to an LLM through an OpenAI-compatible chat completions API (a local
server or a hosted one). An agent is who answers in the chat: a name, the model it runs on and its
instructions (later also the tools it may use). Several agents can share one model.
"""

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, StringConstraints, model_validator

from app import config

SETTINGS_FILE = config.DATA_DIR / "settings.json"

Theme = Literal["system", "light", "dark"]
Text = Annotated[str, StringConstraints(strip_whitespace=True)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]

SIDEBAR_MIN_WIDTH = 240
SIDEBAR_MAX_WIDTH = 600

# The instructions of the Assistant agent made from the earlier single-model setting.
ASSISTANT_INSTRUCTIONS = (
    "Answer from the app's real data: use the tools to look things up instead of guessing. "
    "If a tool fails, say so plainly. Be concrete."
)


class Model(BaseModel):
    id: str
    name: Name  # for people, e.g. "Claude Opus 5.5"
    base_url: Text  # e.g. http://localhost:11434/v1
    model: Text  # the server's name for it, e.g. claude-opus-5-5
    # FOR TESTING ONLY the API key is stored here in plain text; it will move to the OS keychain.
    api_key: Text = ""


class Agent(BaseModel):
    id: str
    name: Name
    model_id: str
    instructions: Text = ""  # its role and rules; app/agent.py adds what babasu is


def _local(href: str) -> str:
    if not href.startswith("/") or href.startswith("//"):
        raise ValueError("a link in the app: a path starting with /, e.g. /views/model-comparison")
    return href


class NavItem(BaseModel):
    """A link added to the navigation (by the agent, e.g. to a view it built)."""

    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
    href: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500), AfterValidator(_local)]


class Settings(BaseModel):
    theme: Theme = "system"
    # Set by dragging the sidebar border and the collapse button, not by the Settings form.
    sidebar_width: int = Field(320, ge=SIDEBAR_MIN_WIDTH, le=SIDEBAR_MAX_WIDTH)
    sidebar_collapsed: bool = False
    # The image editor's click to segment tool, on or off, and its model (segment.MODELS): its
    # toolbar menu, not the Settings form.
    segment_tool: bool = True
    click_model: str = "edgetam/edgetam"
    models: list[Model] = []
    agents: list[Agent] = []
    default_agent: str = ""
    # The chat open in the sidebar (None: a new chat, started by its first message). Set by the
    # chat (New chat, history), not by the Settings form.
    conversation: int | None = None
    # Links added to the navigation, after the app's own sections (config.NAV).
    nav: list[NavItem] = Field([], max_length=30)

    @model_validator(mode="before")
    @classmethod
    def _from_single_model(cls, data):
        """Settings saved before agents had one model in agent_base_url/agent_model/agent_api_key:
        that becomes a model and an Assistant agent using it."""
        if isinstance(data, dict) and data.get("agent_model") and not data.get("models"):
            model_id = slug(data["agent_model"], set())
            data = data | {
                "models": [{"id": model_id, "name": data["agent_model"], "base_url": data.get("agent_base_url", ""),
                            "model": data["agent_model"], "api_key": data.get("agent_api_key", "")}],
                "agents": [{"id": "assistant", "name": "Assistant", "model_id": model_id,
                            "instructions": ASSISTANT_INSTRUCTIONS}],
                "default_agent": "assistant",
            }
        return data

    @model_validator(mode="after")
    def _consistent(self):
        model_ids = {m.id for m in self.models}
        agent_ids = {a.id for a in self.agents}
        if len(model_ids) != len(self.models) or len(agent_ids) != len(self.agents):
            raise ValueError("model and agent ids must be unique")
        for agent in self.agents:
            if agent.model_id not in model_ids:
                raise ValueError(f"agent {agent.name!r} uses a model that doesn't exist")
        if self.agents and self.default_agent not in agent_ids:
            self.default_agent = self.agents[0].id  # the first agent, until one is chosen
        if not self.agents:
            self.default_agent = ""
        return self


def get() -> Settings:
    if SETTINGS_FILE.exists():
        return Settings.model_validate_json(SETTINGS_FILE.read_text())
    return Settings()


def write_file(path: Path, text: str) -> None:
    """Write a file whole: a temporary file, then renamed over it, so it's never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def save(settings: Settings) -> None:
    write_file(SETTINGS_FILE, settings.model_dump_json(indent=2))


def update(**changes) -> Settings:
    """Apply changes, validated by the Settings model, and save. Raises ValidationError."""
    updated = Settings.model_validate(get().model_dump() | changes)
    save(updated)
    return updated


def nav() -> list[dict]:
    """The links added to the navigation: label, href."""
    return [item.model_dump() for item in get().nav]


def set_nav(items: list[dict]) -> list[dict]:
    """Replace the added links (each checked; ValidationError refuses them all) and return them."""
    return [item.model_dump() for item in update(nav=items).nav]


def remove_nav(href: str) -> list[dict]:
    return set_nav([item for item in nav() if item["href"] != href])


# What slug() makes, and what an id in a URL may be (a view's, a saved workflow's).
SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def slug(name: str, taken: set[str]) -> str:
    """A short, readable id from a name ("Qwen 3 8B" -> "qwen-3-8b"), unique among `taken`."""
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "item"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


# --- Reading models and agents (changing them: app/agents.py) ---


def find_agent(agent_id: str | None, s: Settings | None = None) -> tuple[Agent, Model] | None:
    """The agent (the default one if `agent_id` is unknown or empty) and its model; None if none."""
    s = s or get()
    agent = next((a for a in s.agents if a.id == agent_id), None) or next(
        (a for a in s.agents if a.id == s.default_agent), None
    )
    if agent is None:
        return None
    return agent, next(m for m in s.models if m.id == agent.model_id)


def find_model(model_id: str) -> Model | None:
    return next((m for m in get().models if m.id == model_id), None)
