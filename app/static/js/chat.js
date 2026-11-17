// Agent chat in the sidebar. Sends a message and shows the reply as it streams: text as it's
// written, each tool call with its state (running, done, error), and how the turn ended. The
// server saves everything, so the next page renders the same conversation (partials/chat.html).
// The reply doesn't depend on this page: a page opened while the agent is answering follows the
// reply from its first event ([data-following]), and stop is a request of its own.
// The agent's text is markdown, shown formatted (js/markdown.js): saved parts once the page
// loads, a streaming part again as it grows.
// Markup comes from the <template>s in that partial; this file only fills them in.
import { reason } from "./background-form.js";
import { render } from "./markdown.js";

const chat = document.querySelector(".chat");

if (chat) {
  const form = chat.querySelector(".chat__form");
  const input = form.elements.message;
  const messages = chat.querySelector(".chat__messages");
  const empty = chat.querySelector(".chat__empty");
  let leaving = false; // the page is going away: its stream is cut, the reply goes on
  addEventListener("pagehide", () => { leaving = true; });

  const clone = (id) => chat.querySelector(`#${id}`).content.firstElementChild.cloneNode(true);
  const atBottom = () => messages.scrollHeight - messages.scrollTop - messages.clientHeight < 24;

  // Follow the reply only while the reader is at the bottom; scrolling up to read stops it.
  function append(parent, element) {
    const follow = atBottom();
    parent.append(element);
    if (follow) messages.scrollTop = messages.scrollHeight;
    return element;
  }

  // One agent turn, built from the stream's events: a new one, or the one the page rendered.
  function agentTurn(turn = append(messages, clone("chat-agent"))) {
    // "Thinking…" while the agent works
    let status = turn.querySelector(".chat__note--status") ?? append(turn, clone("chat-status"));
    let part = null; // the text part being written: { element, markdown }; a tool call ends it
    let frame = 0; // a render of that part waiting for the next frame
    let byLine = null; // who answers; added under everything else when the turn ends
    const tools = new Map();
    let running = {}; // what the running tool shows under it: { output, bar }

    function clearStatus() {
      status?.remove();
      status = null;
    }

    // Render the part's markdown so far (at most once a frame, however fast deltas come).
    function draw() {
      cancelAnimationFrame(frame);
      frame = 0;
      const follow = atBottom();
      part.element.innerHTML = render(part.markdown);
      if (follow) messages.scrollTop = messages.scrollHeight;
    }

    return {
      text(delta) {
        clearStatus();
        part ??= { element: append(turn, clone("chat-text")), markdown: "" };
        part.markdown += delta;
        frame ||= requestAnimationFrame(draw);
      },
      tool({ id, label, note, images, status: state }) {
        if (frame) draw(); // the part ends as written
        part = null;
        clearStatus();
        let element = tools.get(id);
        if (!element) {
          element = append(turn, clone("chat-tool"));
          element.querySelector(".chat__tool-label").textContent = label;
          tools.set(id, element);
        }
        element.querySelector(".chat__tool-note").textContent = note ?? "";
        element.dataset.status = state;
        // The images it found, drawn by the server (partials/chat-images.html) as in a saved turn.
        if (images) {
          const query = new URLSearchParams({ project_id: images.project_id });
          for (const mediaId of images.media_ids) query.append("media_ids", mediaId);
          fetch(`/agent/images?${query}`)
            .then((response) => (response.ok ? response.text() : ""))
            .then((html) => element.insertAdjacentHTML("beforeend", html));
        }
        running = {};
        // After a tool and between rounds, show that the agent is still working.
        if (state !== "running") status = append(turn, clone("chat-status"));
      },
      // What the running tool prints (appended, following its end) and how far it is.
      output({ text }) {
        running.output ??= append(turn, clone("chat-output"));
        running.output.textContent += text;
        running.output.scrollTop = running.output.scrollHeight;
      },
      progress({ done, total }) {
        running.bar ??= append(turn, clone("chat-progress"));
        const value = total ? Math.min(done / total, 1) : 0;
        running.bar.style.setProperty("--value", value);
        running.bar.setAttribute("aria-valuenow", Math.round(value * 100));
      },
      by(agent, model) {
        const line = clone("chat-by");
        line.textContent = agent;
        line.title = `${agent} · ${model}`;
        byLine = line;
      },
      end(state, message) {
        if (frame) draw();
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

  // While a reply streams (also as the server renders a page mid-reply): the button is "stop".
  const busy = () => chat.classList.contains("chat--busy");

  function setBusy(yes) {
    chat.classList.toggle("chat--busy", yes);
    if (yes) messages.setAttribute("aria-busy", "true");
    else messages.removeAttribute("aria-busy");
  }

  // The page hears what the agent does, as it does it, so it can show what's being made (the
  // workflow builder draws a workflow as it's written); the chat doesn't know which page does:
  //   "agent:call"  {id, name, arguments}  a tool call as the model writes it: what's complete of
  //                                        its arguments so far (an object), at most once a frame
  //   "agent:tool"  {id, label, status, note, images}  a tool call running, then done or refused ("error")
  //   "agent:end"                          the reply is over (before the page reloads)
  const hear = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));

  // The value of the longest complete prefix of JSON text being written (a tool call's arguments),
  // closed as it stands (a partly written string, number or key is left out); null if none yet.
  function partialJson(text) {
    const closers = [];
    let inString = false, escaped = false, cut = -1, open = [];
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (c === "\\") escaped = true;
        else if (c === '"') inString = false;
      } else if (c === '"') inString = true;
      else if (c === "{" || c === "[") closers.push(c === "{" ? "}" : "]");
      else if (c === "}" || c === "]") { closers.pop(); cut = i + 1; open = [...closers]; }
      else if (c === ",") { cut = i; open = [...closers]; }
    }
    if (cut < 0) return null;
    try {
      return JSON.parse(text.slice(0, cut) + open.reverse().join(""));
    } catch {
      return null;
    }
  }

  // Show a reply as it streams into `turn`: newline-delimited JSON events (app/agent.py,
  // run_turn), from the request that sent the message or one that follows the reply.
  async function follow(request, turn) {
    setBusy(true);
    let ended = false;
    let open = null; // a page the agent wants to show; opened when the reply is done
    let changed = false; // the agent changed what pages show: this one reloads when the reply is done
    const calls = new Map(); // the tool calls being written: id -> {name, text: their arguments so far}
    const written = new Set(); // the ids written to since the page last heard them
    // The page hears each call written to, once a frame (a model writes faster than it's worth
    // showing), and before anything that follows it (its tool running, the end of the reply).
    function heard() {
      for (const id of written) {
        const { name, text } = calls.get(id);
        const value = partialJson(text);
        if (value) hear("agent:call", { id, name, arguments: value });
      }
      written.clear();
    }
    try {
      const response = await request;
      // Nothing to follow: the app restarted since this page was made, which stopped the reply.
      if (response.status === 204) return turn.end("stopped");
      if (!response.ok) throw new Error(await reason(response));
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
          else if (event.type === "tool") {
            heard();
            turn.tool(event);
            hear("agent:tool", event);
          } else if (event.type === "call") {
            const call = calls.get(event.id) ?? { name: event.name, text: "" };
            call.text += event.delta;
            calls.set(event.id, call);
            if (!written.size) requestAnimationFrame(heard);
            written.add(event.id);
          }
          else if (event.type === "output") turn.output(event);
          else if (event.type === "progress") turn.progress(event);
          else if (event.type === "open") open = event.url;
          else if (event.type === "changed") changed = true;
          else if (event.type === "done") (turn.end("done"), (ended = true));
          else if (event.type === "stopped") (turn.end("stopped"), (ended = true));
          else if (event.type === "error") (turn.end("error", event.message), (ended = true));
        }
      }
      if (!ended) turn.end("error", "The reply ended unexpectedly.");
      heard();
      hear("agent:end"); // a page may keep what it drew as unsaved work: it's then not reloaded
      // The chat is saved, so it's there on the new page. A page showing that it has unsaved work
      // ([data-unsaved], e.g. the workflow builder's note) isn't reloaded under the person.
      if (open) location.assign(open);
      else if (changed && !document.querySelector("[data-unsaved]:not([hidden])")) location.reload();
    } catch (error) {
      if (!leaving) turn.end("error", error.message || "The app didn't answer.");
      heard();
      hear("agent:end");
    } finally {
      setBusy(false);
    }
  }

  function send(text) {
    empty.hidden = true;
    const user = clone("chat-user");
    user.textContent = text;
    append(messages, user);
    messages.scrollTop = messages.scrollHeight; // your own message always comes into view
    const request = fetch("/api/agent/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Where it was written: a chat that starts with this message reopens here.
      body: JSON.stringify({
        text, page: location.pathname + location.search, page_title: document.title,
        about: document.querySelector('meta[name="about"]')?.content ?? "",
      }),
    });
    follow(request, agentTurn());
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
    if (busy()) {
      // The button is "stop" while a reply streams; the reply ends with a "stopped" event.
      fetch("/api/agent/stop", { method: "POST" });
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

  // Saved text parts hold their markdown as text (partials/chat.html).
  for (const part of messages.querySelectorAll(".chat__text")) part.innerHTML = render(part.textContent);
  messages.scrollTop = messages.scrollHeight; // open at the latest message

  // The agent is still answering (the page was opened mid-reply): follow the reply.
  const following = messages.querySelector("[data-following]");
  if (following) follow(fetch("/api/agent/turn"), agentTurn(following));
}
