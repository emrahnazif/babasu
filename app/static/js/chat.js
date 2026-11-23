// Agent chat in the sidebar. Sends a message and shows the reply as it streams: what the model
// thinks, text as it's written, each tool call from its first piece (as it's written, running,
// then done or failed), and how the turn ended. The server saves everything, so the next page
// renders the same conversation (partials/chat.html).
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
  // Images drawn by the server as in a saved turn (partials/chat-images.html), with their objects:
  // those a step found, or those dropped with a message.
  const html = (url) => fetch(url).then((response) => (response.ok ? response.text() : ""));
  const drawn = (mediaIds) => html(`/agent/images?${new URLSearchParams(mediaIds.map((id) => ["media_ids", id]))}`);
  const atBottom = () => messages.scrollHeight - messages.scrollTop - messages.clientHeight < 24;

  // Change the chat, following the reply only while the reader is at the bottom: scrolling up to
  // read stops it.
  function grow(change) {
    const follow = atBottom();
    change();
    if (follow) messages.scrollTop = messages.scrollHeight;
  }

  function append(parent, element) {
    grow(() => parent.append(element));
    return element;
  }

  // How long something took, at a glance: "12s", "1m 5s" (formatting.py's took writes the same).
  function took(seconds) {
    const whole = Math.max(Math.round(seconds), 1);
    return whole >= 60 ? `${Math.floor(whole / 60)}m ${whole % 60}s` : `${whole}s`;
  }

  // A line's state (partials/chat.html line()): its icon for it, and its title and detail if given.
  function setLine(line, state, title, detail) {
    line.dataset.status = state;
    line.querySelector(".chat__line-icon").replaceChildren(clone(`chat-icon-${state}`));
    if (title != null) line.querySelector(".chat__line-title").textContent = title;
    if (detail != null) line.querySelector(".chat__line-detail").textContent = detail;
  }

  // Add to a window that shows the last lines of what's written (thinking, a step's arguments and
  // output): kept at its end, and faded at the top once it's full ([data-cut]). What's written in a
  // frame is added at once, however fast it comes.
  const unwritten = new Map(); // window -> text not added yet
  function write(window, text) {
    if (!unwritten.size) requestAnimationFrame(() => grow(() => {
      for (const [shown, more] of unwritten) {
        shown.append(more);
        shown.scrollTop = shown.scrollHeight;
        if (!shown.hasAttribute("data-cut") && shown.scrollHeight > shown.clientHeight) shown.toggleAttribute("data-cut", true);
      }
      unwritten.clear();
    }));
    unwritten.set(window, (unwritten.get(window) ?? "") + text);
  }

  // Fold or unfold what folds as the turn goes, unless the person has opened or closed it.
  const fold = (details, open) => { if (!("touched" in details.dataset)) details.open = open; };
  messages.addEventListener("click", (event) => {
    const summary = event.target.closest("summary");
    if (summary) summary.parentElement.dataset.touched = "";
  });

  // One agent turn, built from the stream's events: a new one, or the one the page rendered. Its
  // parts as partials/chat.html draws a saved one: what its model thinks, its text, each tool call.
  function agentTurn(turn = append(messages, clone("chat-agent"))) {
    // "Working" while nothing else shows that it is (waiting for the model, storing what was dropped).
    let status = turn.querySelector(".chat__status") ?? append(turn, clone("chat-status"));
    let part = null; // the text part being written: { element, markdown }; anything else ends it
    let frame = 0; // a render of that part waiting for the next frame
    let byLine = null; // who answers; added under everything else when the turn ends
    let who = ""; // the agent and its model, the by-line's title
    const tokens = { read: 0, written: 0 }; // what the model's calls in this turn read and wrote
    const thoughts = new Map(); // id -> { element, line, text, since, timer }
    const steps = new Map(); // tool call id -> { element, line, body, arguments, output, bar }
    let running = null; // the step whose tool is running: what it prints goes there

    function clearStatus() {
      status?.remove();
      status = null;
    }

    // Render the text part's markdown so far (at most once a frame, however fast deltas come).
    function draw() {
      cancelAnimationFrame(frame);
      frame = 0;
      grow(() => { part.element.innerHTML = render(part.markdown); });
    }

    function endText() {
      if (frame) draw();
      part = null;
    }

    // A part other than text (thinking, a tool call): it ends the text part being written.
    function added(element) {
      endText();
      clearStatus();
      return append(turn, element);
    }

    // Done thinking: for how long (as the server says; not known if the reply was cut off).
    function endThought(thought, seconds) {
      clearInterval(thought.timer);
      setLine(thought.line, "done", seconds == null ? "Thought" : `Thought for ${took(seconds)}`, "");
      fold(thought.element, false);
    }

    // A tool call's step: shown open from its first piece, what it does said as it's known.
    function step(id, label) {
      let found = steps.get(id);
      if (!found) {
        const element = added(clone("chat-tool"));
        found = { element, line: element.querySelector(".chat__line"), body: element.querySelector(".chat__step-body") };
        element.open = true;
        steps.set(id, found);
      }
      if (label) setLine(found.line, found.line.dataset.status, label);
      return found;
    }

    return {
      // Storing the images the person attached, before the agent starts: how far it is.
      storing({ done, total }) {
        const title = status.querySelector(".chat__line-title");
        if (total == null) title.textContent = "Storing the images";
        else if (done < total) title.textContent = `Storing the images: ${done.toLocaleString("en")} of ${total.toLocaleString("en")}`;
        else status.replaceWith((status = clone("chat-status")));
      },
      // Those images, under the person's message (the first few); in place of those the page
      // drew if it was opened while they were being stored.
      attached(messageId) {
        html(`/agent/attached/${messageId}`).then((images) => {
          const drawnBefore = turn.previousElementSibling;
          if (drawnBefore?.classList.contains("chat__images")) drawnBefore.outerHTML = images;
          else turn.insertAdjacentHTML("beforebegin", images);
        });
      },
      // What the model thinks, open while it streams, with its time so far.
      reasoning({ id, since, delta }) {
        let thought = thoughts.get(id);
        if (!thought) {
          const element = added(clone("chat-thinking"));
          const line = element.querySelector(".chat__line");
          const detail = line.querySelector(".chat__line-detail");
          const tick = () => { detail.textContent = took(Date.now() / 1000 - since); };
          thought = { element, line, text: element.querySelector(".chat__thinking-text"), since, timer: setInterval(tick, 1000) };
          tick();
          thoughts.set(id, thought);
        }
        write(thought.text, delta);
      },
      thought({ id, seconds }) {
        const thought = thoughts.get(id);
        if (thought) endThought(thought, seconds);
      },
      text(delta) {
        clearStatus();
        part ??= { element: append(turn, clone("chat-text")), markdown: "" };
        part.markdown += delta;
        frame ||= requestAnimationFrame(draw);
      },
      // A tool call as the model writes it: shown from its first piece, its arguments as they come.
      call({ id, label, delta }) {
        const found = step(id, label);
        if (!delta) return;
        found.arguments ??= append(found.body, clone("chat-arguments"));
        write(found.arguments, delta);
      },
      // The call running (open, with what it prints), then done or failed: folded to what came out,
      // open on what there is to see (why it failed, the images it found, the picture it looked at).
      tool({ id, label, note, error, images, picture, status: state }) {
        const found = step(id, label);
        clearStatus(); // a "Working" shown after the call before it
        setLine(found.line, state, null, note ?? "");
        if (state === "running") {
          running = found;
          fold(found.element, true);
          return;
        }
        running = null;
        for (const shown of [found.arguments, found.output, found.bar]) shown?.remove();
        if (error) append(found.body, clone("chat-step-error")).textContent = error;
        if (images) drawn(images).then((html) => grow(() => found.body.insertAdjacentHTML("beforeend", html)));
        if (picture) append(found.body, clone("chat-picture")).querySelector("img").src = picture;
        fold(found.element, Boolean(error || images || picture));
        // Between rounds, show that it's still working.
        status = append(turn, clone("chat-status"));
      },
      // What the running tool prints (appended, following its end) and how far it is.
      output({ text }) {
        if (!running) return;
        running.output ??= append(running.body, clone("chat-output"));
        write(running.output, text);
      },
      progress({ done, total }) {
        if (!running) return;
        running.bar ??= append(running.body, clone("chat-progress"));
        const value = total ? Math.min(done / total, 1) : 0;
        running.bar.style.setProperty("--value", value);
        running.bar.setAttribute("aria-valuenow", Math.round(value * 100));
      },
      by(agent, model) {
        byLine = clone("chat-by");
        byLine.prepend(agent);
        byLine.title = who = `${agent} · ${model}`;
      },
      usage({ tokens_in, tokens_out }) {
        tokens.read += tokens_in;
        tokens.written += tokens_out;
        byLine.title = `${who} · ${tokens.read.toLocaleString("en")} tokens read, ${tokens.written.toLocaleString("en")} written`;
      },
      end(state, message, seconds) {
        endText();
        clearStatus();
        // Thinking the server didn't say the end of (the reply was cut off) ends here.
        for (const thought of thoughts.values()) {
          if (thought.line.dataset.status === "running") endThought(thought);
        }
        if (state !== "done") {
          const note = clone(state === "error" ? "chat-error" : "chat-stopped");
          if (message) note.textContent = message;
          append(turn, note);
        }
        if (byLine) {
          if (seconds != null) byLine.querySelector(".chat__took").textContent = ` · ${took(seconds)}`;
          append(turn, byLine);
        }
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
          if (event.type === "storing") turn.storing(event);
          else if (event.type === "into") into(event.url);
          else if (event.type === "attached") turn.attached(event.message_id);
          else if (event.type === "agent") turn.by(event.agent, event.model);
          else if (event.type === "reasoning") turn.reasoning(event);
          else if (event.type === "thought") turn.thought(event);
          else if (event.type === "text") turn.text(event.delta);
          else if (event.type === "tool") {
            heard();
            turn.tool(event);
            hear("agent:tool", event);
          } else if (event.type === "call") {
            turn.call(event);
            const call = calls.get(event.id) ?? { name: event.name, text: "" };
            call.text += event.delta;
            calls.set(event.id, call);
            if (!written.size) requestAnimationFrame(heard);
            written.add(event.id);
          }
          else if (event.type === "output") turn.output(event);
          else if (event.type === "progress") turn.progress(event);
          else if (event.type === "usage") turn.usage(event);
          else if (event.type === "open") open = event.url;
          else if (event.type === "changed") changed = true;
          else if (["done", "stopped", "error"].includes(event.type)) (turn.end(event.type, event.message, event.seconds), (ended = true));
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

  // The project made for the images sent with a message: the chat goes into it at once (the reply
  // goes on, followed there), unless this page has unsaved work or is that project's already.
  function into(url) {
    if (location.pathname !== url && !document.querySelector("[data-unsaved]:not([hidden])")) location.assign(url);
  }

  // The message, with the images attached to it (a drop zone's paths, in this form wherever it is).
  function send(text, paths) {
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
        text, paths, page: location.pathname + location.search, page_title: document.title,
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
    if (event.submitter?.hasAttribute("formaction")) return; // sent elsewhere (a drop zone's action): not a message
    event.preventDefault();
    if (busy()) {
      // The button is "stop" while a reply streams; the reply ends with a "stopped" event.
      fetch("/api/agent/stop", { method: "POST" });
      return;
    }
    const text = input.value.trim();
    if (!text) return;
    const paths = new FormData(form).getAll("paths");
    form.reset(); // empties the field and what's attached
    send(text, paths);
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
