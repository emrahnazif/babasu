// Agent chat in the sidebar. Sends a message and shows the reply as it streams: text as it's
// written, each tool call with its state (running, done, error), and how the turn ended. The
// server saves everything, so the next page renders the same conversation (partials/chat.html).
// Markup comes from the <template>s in that partial; this file only fills them in.
const chat = document.querySelector(".chat");

if (chat) {
  const form = chat.querySelector(".chat__form");
  const input = form.elements.message;
  const messages = chat.querySelector(".chat__messages");
  const empty = chat.querySelector(".chat__empty");
  let controller = null; // set while a reply streams; aborting it is "stop"

  const clone = (id) => chat.querySelector(`#${id}`).content.firstElementChild.cloneNode(true);
  const atBottom = () => messages.scrollHeight - messages.scrollTop - messages.clientHeight < 24;

  // Follow the reply only while the reader is at the bottom; scrolling up to read stops it.
  function append(parent, element) {
    const follow = atBottom();
    parent.append(element);
    if (follow) messages.scrollTop = messages.scrollHeight;
    return element;
  }

  // One agent turn, built from the stream's events.
  function agentTurn() {
    const turn = append(messages, clone("chat-agent"));
    let status = append(turn, clone("chat-status")); // "Thinking…" while the agent works
    let text = null; // the text part being written; a tool call ends it
    let byLine = null; // who answers; added under everything else when the turn ends
    const tools = new Map();

    function clearStatus() {
      status?.remove();
      status = null;
    }

    return {
      text(delta) {
        clearStatus();
        text ??= append(turn, clone("chat-text"));
        const follow = atBottom();
        text.append(delta); // a new text node, not a rewrite of everything so far
        if (follow) messages.scrollTop = messages.scrollHeight;
      },
      tool({ id, name, status: state }) {
        text = null;
        clearStatus();
        let element = tools.get(id);
        if (!element) {
          element = append(turn, clone("chat-tool"));
          element.querySelector(".chat__tool-name").textContent = name;
          tools.set(id, element);
        }
        element.dataset.status = state;
        // After a tool and between rounds, show that the agent is still working.
        if (state !== "running") status = append(turn, clone("chat-status"));
      },
      by(agent, model) {
        const line = clone("chat-by");
        line.textContent = agent;
        line.title = `${agent} · ${model}`;
        byLine = line;
      },
      end(state, message) {
        clearStatus();
        turn.dataset.status = state;
        if (state !== "done") {
          const note = clone(state === "error" ? "chat-error" : "chat-stopped");
          if (message) note.textContent = message;
          append(turn, note);
        }
        if (byLine) append(turn, byLine);
      },
    };
  }

  async function send(text) {
    empty.hidden = true;
    const user = clone("chat-user");
    user.textContent = text;
    append(messages, user);
    messages.scrollTop = messages.scrollHeight; // your own message always comes into view

    const turn = agentTurn();
    controller = new AbortController();
    chat.classList.add("chat--busy");
    messages.setAttribute("aria-busy", "true");
    let ended = false;
    try {
      const response = await fetch("/api/agent/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`The server answered ${response.status}.`);
      // Newline-delimited JSON: one event per line (app/agent.py, run_turn).
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines.filter(Boolean)) {
          const event = JSON.parse(line);
          if (event.type === "agent") turn.by(event.agent, event.model);
          else if (event.type === "text") turn.text(event.delta);
          else if (event.type === "tool") turn.tool(event);
          else if (event.type === "done") (turn.end("done"), (ended = true));
          else if (event.type === "error") (turn.end("error", event.message), (ended = true));
        }
      }
      if (!ended) turn.end("error", "The reply ended unexpectedly.");
    } catch (error) {
      if (error.name === "AbortError") turn.end("stopped");
      else turn.end("error", error.message || "The app didn't answer.");
    } finally {
      controller = null;
      chat.classList.remove("chat--busy");
      messages.removeAttribute("aria-busy");
    }
  }

  // Enter sends, Shift+Enter adds a new line (isComposing: don't send mid-IME input).
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (controller) {
      controller.abort(); // the button is "stop" while a reply streams
      return;
    }
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    send(text);
  });

  // Switch this chat's agent; it answers from the next message on.
  chat.querySelector(".chat__agent")?.addEventListener("change", (event) => {
    fetch("/api/agent/conversation", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: event.target.value }),
    });
  });

  messages.scrollTop = messages.scrollHeight; // open at the latest message
}
