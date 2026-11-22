"""How the agent talks to the model's server."""

from app import agent, settings


def test_a_model_without_a_key_sends_the_placeholder_keyless_servers_accept():
    """Unsloth Studio's keyless access refuses an arbitrary bearer ("none") but takes "not-needed"."""
    model = settings.Model(id="local", name="Local", base_url="http://localhost:8888/v1", model="m")
    assert agent.client(model).api_key == "not-needed"
    assert agent.client(model.model_copy(update={"api_key": "sk-mine"})).api_key == "sk-mine"
