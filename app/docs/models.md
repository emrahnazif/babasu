# Models
The vision models (downloaded, loaded, licences) and the language models and agents the chat runs on.

## Vision models
- mozo publishes the vision models: families (SAM 3, Grounding DINO, EdgeTAM, YOLO, CLIP, …), each with variants of different sizes and licences.
- **Downloading** stores a model's weights in the workspace, as a background job (io lane). Removing deletes them.
- **Loading** puts a downloaded model in memory (gpu lane); workflows load what they need themselves. **Unloading** frees the memory.
- Licences differ (MIT, Apache-2.0, the SAM License, AGPL…): say a model's licence when suggesting it.
- Workflows and auto-annotators list the models they need and how much is left to download.

## Language models and agents
- The chat runs on an agent: a language model (any OpenAI-compatible server: local like Ollama, or a hosted one) with instructions. Settings → Models sets up the models (address, key, model name); Settings → Agents the agents and the default one.
- The chat's agent picker switches the agent for the conversation.
