// Click to segment, in the annotation editor (js/annotation-editor.js): with the tool on
// (#click-tool, aria-pressed) and a model in use (its data-in-use), the object under the pointer is
// shown (MarkinJS's preview), found by the click model through the canvas's data-segment URL.
//   - A click fixes it as a candidate; a drag (MarkinJS's canvasdrag) asks for what's in its box.
//   - Then every click shapes it: inside it leaves that part out, outside it adds to it
//     (Option/Alt flips that); another drag replaces the box; Backspace takes back the last point
//     (then the box). Enter or Add keeps it as an annotation of the current class; Esc or Discard
//     drops it.
//   - A selected object can be refined (R, or Refine next to it): it becomes the candidate, its box
//     the prompt, and Add replaces it.
//   - The tool's menu picks the model ([data-click-model]; one not downloaded downloads, another
//     answers meanwhile) and turns it off and on ([data-click-toggle], S): PUT /api/click-tool.
// Off, the pointer sends nothing, and a drag on the image moves the view again (with the tool on,
// Space + drag does).
//
// Getting ready, said over the image (#click-busy): the image is encoded ahead (prepare), then
// the next one, so → opens a ready image. A model not in memory is loaded by its load job, never
// by the call: prepare answers 409 with the job, the pill says the model is loading (or waiting
// for the job ahead of it), and prepare is asked again when a job ends.
//
// Only the latest request matters: a new one aborts the one on its way, and the server skips calls
// a newer one overtook (204).
import { toast } from "./toast.js";

const HOVER_PAUSE = 30; // ms the pointer rests before the object under it is asked for

export function clickTool(editor, canvas, currentClass) {
  const actions = document.querySelector("#candidate-actions");
  const pointsLayer = document.querySelector("#candidate-points");
  const refineAction = document.querySelector("#refine-action");
  const button = document.querySelector("#click-tool");
  const toggle = document.querySelector("[data-click-toggle]");
  const pressed = () => button.getAttribute("aria-pressed") === "true";
  const on = () => pressed() && button.dataset.inUse !== "";

  // Once clicked: { points, labels, box, shape (the server's annotation), refining (the annotation
  // it replaces, if any) }.
  let candidate = null;
  let hovered = null; // the shape under the pointer, before a click
  let hoverTimer = null;
  let inFlight = null; // AbortController of the request on its way
  let altPressed = false; // Option/Alt held on the press that makes the next click

  let dragging = false; // a box is being drawn

  // --- Getting ready: the model in memory, then this image's encoding, then the next one's ---

  const busy = document.querySelector("#click-busy");
  const say = (text) => ((busy.textContent = text ?? ""), (busy.hidden = !text));
  const modelName = () => document.querySelector(`[data-click-model="${button.dataset.inUse}"]`)?.dataset.name ?? "The click model";
  let loading = false; // its load job is under way: prepare again when a job ends
  let preparing = null; // AbortController of the prepare on its way: a newer one replaces it
  const post = (url, signal) => fetch(url, { method: "POST", signal });

  async function prepare() {
    preparing?.abort();
    const controller = (preparing = new AbortController());
    say(button.dataset.loaded === "true" ? "Getting the image ready…" : `Loading ${modelName()}…`);
    try {
      const response = await post(canvas.dataset.prepare, controller.signal);
      const answer = response.status === 409 ? await response.json() : null;
      if (answer?.loading) {
        const { model, ahead, started } = answer.loading;
        loading = true;
        say(ahead ? `Waiting for ${ahead}, then loading ${model}…` : `Loading ${model}…`);
        if (started) document.dispatchEvent(new CustomEvent("submitted")); // the jobs list shows it now
        return;
      }
      loading = false;
      say(null);
      if (!response.ok) throw new Error(answer?.detail ?? response.statusText);
      button.dataset.loaded = "true";
      if (canvas.dataset.prepareNext) post(canvas.dataset.prepareNext).catch(() => {});
    } catch (error) {
      if (error.name === "AbortError") return; // a newer prepare took over
      say(null);
      toast("Couldn't get the image ready", error.message);
    } finally {
      if (preparing === controller) preparing = null;
    }
  }

  const takeOver = () => editor.setOptions({ panOnEmpty: !on() }); // a drag is a box while it's on
  takeOver();
  if (on()) prepare();

  // The object at these points, or null if nothing's there; undefined if this answer doesn't count
  // (overtaken by a newer request, or failed: said once in a toast).
  async function ask(points, labels, box = null) {
    inFlight?.abort();
    const controller = (inFlight = new AbortController());
    try {
      const response = await fetch(canvas.dataset.segment, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points, labels, box, class_id: classOf() }),
        signal: controller.signal,
      });
      if (response.status === 204) return undefined; // overtaken by a newer call
      if (!response.ok) {
        const { detail } = await response.json();
        throw new Error(typeof detail === "string" ? detail : response.statusText);
      }
      return await response.json();
    } catch (error) {
      if (error.name !== "AbortError") toast("Couldn't segment", error.message);
      return undefined;
    } finally {
      if (inFlight === controller) inFlight = null;
    }
  }

  const classOf = () => candidate?.refining?.class_id ?? currentClass();
  const preview = (shape) => editor.setPreview(shape && { ...shape, class_id: classOf() });

  // --- Before a click: the object under the pointer ---

  function clearHover() {
    clearTimeout(hoverTimer);
    inFlight?.abort();
    hovered = null;
    if (!candidate) preview(null);
  }

  editor.on("hover", ({ position, id }) => {
    if (candidate || dragging || !on()) return;
    clearTimeout(hoverTimer);
    if (id != null) return clearHover(); // over an annotation: that one is edited as usual
    hoverTimer = setTimeout(async () => {
      const shape = await ask([[position.x, position.y]], [1]);
      if (shape === undefined || candidate || dragging || !on()) return;
      hovered = shape;
      preview(shape);
    }, HOVER_PAUSE);
  });
  canvas.addEventListener("pointerleave", () => candidate || clearHover());

  // --- After a click: shaping the candidate ---

  document.addEventListener("pointerdown", (event) => { altPressed = event.altKey; }, true);

  function start(points, labels, box, shape, refining = null) {
    clearTimeout(hoverTimer);
    candidate = { points, labels, box, shape, refining };
    editor.disable(); // every click is the candidate's now, also on another annotation
    editor.deselect();
  }

  editor.on("canvasclick", async ({ position }) => {
    if (!on()) return;
    const point = [position.x, position.y];
    if (!candidate) start([point], [1], null, hovered);
    else {
      const inside = candidate.shape && contains(candidate.shape, point);
      const label = inside !== altPressed ? 0 : 1;
      // Adding outside the box says the object goes beyond it: the box goes, and a point inside
      // what it found keeps that part.
      if (label && candidate.box && !inBox(candidate.box, point)) {
        if (candidate.shape) {
          candidate.points.unshift(pointIn(candidate.shape));
          candidate.labels.unshift(1);
        }
        candidate.box = null;
      }
      candidate.points.push(point);
      candidate.labels.push(label);
    }
    await reshape();
  });

  async function reshape() {
    place();
    const asked = candidate;
    const shape = await ask(asked.points, asked.labels, asked.box);
    if (shape === undefined || candidate !== asked) return;
    candidate.shape = shape;
    preview(shape);
    place();
  }

  function finish() {
    const refined = candidate?.refining;
    candidate = null;
    hovered = null;
    inFlight?.abort();
    preview(null);
    place();
    editor.enable();
    if (refined) editor.select(refined.id); // back to the object, as it was or refined
  }

  function accept() {
    if (!candidate?.shape) return;
    const { score, ...shape } = candidate.shape; // a person kept it: not a model's guess anymore
    const { refining } = candidate;
    const classId = classOf();
    finish();
    if (!refining) editor.createAnnotation({ ...shape, class_id: classId });
    // A detection project's answer is a box: an outline the object had stays.
    else editor.updateAnnotation(refining.id, { ...shape, class_id: classId, score: null });
  }

  // --- Refine: a selected object becomes the candidate again ---

  let selected = null; // the selected annotation, while the tool is on and nothing is being shaped

  const placeRefine = () => placeBeside(refineAction, on() && !candidate && selected?.bbox);
  editor.on("select", ({ annotation }) => { selected = annotation; placeRefine(); });
  editor.on("deselect", () => { selected = null; placeRefine(); });
  editor.on("dragstart", () => { refineAction.hidden = true; });
  editor.on("change", () => {
    if (selected) selected = editor.getAnnotation(selected.id);
    placeRefine();
  });

  refineAction.querySelector("[data-refine]").addEventListener("click", () => {
    if (!on() || candidate || !selected?.bbox) return;
    const [x, y, w, h] = selected.bbox;
    const refining = selected;
    start([], [], [x, y, x + w, y + h], refining, refining);
    placeRefine();
    reshape();
  });

  document.addEventListener("keydown", (event) => {
    // A key the editor took (Escape ending a box drag) isn't ours.
    if (!candidate || event.defaultPrevented || event.target.closest("input, textarea, select, [contenteditable]")) return;
    if (event.key === "Enter") accept();
    else if (event.key === "Escape") finish();
    else if (event.key === "Backspace") {
      if (candidate.points.length) {
        candidate.points.pop();
        candidate.labels.pop();
      } else candidate.box = null;
      if (candidate.points.length || candidate.box) reshape();
      else finish();
    } else return;
    event.preventDefault();
  });
  actions.querySelector("[data-accept]").addEventListener("click", accept);
  actions.querySelector("[data-reject]").addEventListener("click", finish);

  // --- A drag: the object in a box (MarkinJS reports a drag on empty canvas; not a click) ---

  editor.on("canvasdrag", ({ box }) => {
    if (!on()) return;
    if (!dragging) clearHover();
    dragging = true;
    editor.setPreview({ class_id: classOf(), bbox: box });
  });
  editor.on("canvasdragend", async ({ box: [x, y, w, h], cancelled }) => {
    dragging = false;
    if (!on()) return;
    if (cancelled || !w || !h) return preview(candidate?.shape ?? null);
    const box = [x, y, x + w, y + h]; // the segment API's x1, y1, x2, y2
    if (!candidate) start([], [], box, null);
    else candidate.box = box;
    await reshape();
  });

  // The candidate in the class picked meanwhile.
  document.querySelector("#class-menu").addEventListener("click", (event) => {
    if (!candidate || !event.target.closest("[data-class-id]")) return;
    if (candidate.refining) candidate.refining = { ...candidate.refining, class_id: currentClass() };
    preview(candidate.shape);
  });

  // --- Points and Add/Discard, placed on screen over the image ---

  function place() {
    pointsLayer.replaceChildren(
      ...(candidate?.points ?? []).map(([x, y], i) => {
        const dot = document.createElement("span");
        dot.className = candidate.labels[i] ? "dot" : "dot exclude";
        const at = editor.toClient(x, y);
        dot.style.left = `${at.x}px`;
        dot.style.top = `${at.y}px`;
        return dot;
      }),
    );
    placeBeside(actions, candidate?.shape?.bbox);
  }

  // Show a floating element beside a box's top-right corner, inside the window; hide it without one.
  function placeBeside(element, bbox) {
    element.hidden = !bbox;
    if (!bbox) return;
    const corner = editor.toClient(bbox[0] + bbox[2], bbox[1]);
    element.style.left = `${Math.min(corner.x + 8, innerWidth - element.offsetWidth - 8)}px`;
    element.style.top = `${Math.max(corner.y, 8)}px`;
  }
  editor.on("viewchange", () => (candidate ? place() : placeRefine()));

  // --- The menu: the model, off and on ---

  // Show the tool as the server has it ({on, in_use, …}): pressed, the model answering, the
  // toggle's words; prepare the image again for a model that's new, stop for one that's off.
  function show(tool) {
    const before = on() && button.dataset.inUse;
    button.setAttribute("aria-pressed", String(tool.on));
    button.dataset.inUse = tool.in_use ?? "";
    button.dataset.loaded = String(tool.loaded);
    toggle.lastChild.textContent = tool.on ? "Turn off" : "Turn on";
    takeOver();
    if (!on()) {
      clearHover();
      if (candidate) finish();
      loading = false;
      say(null);
    } else if (before !== tool.in_use || loading) {
      clearHover();
      prepare();
    }
    placeRefine();
  }

  async function change(body) {
    try {
      const response = await fetch("/api/click-tool", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(response.statusText);
      show(await response.json());
      document.dispatchEvent(new CustomEvent("click-tool-changed")); // the models' list again
    } catch (error) {
      toast("Couldn't change the click tool", error.message);
    }
  }

  document.addEventListener("click", (event) => {
    const item = event.target.closest("[data-click-model]");
    if (item) change({ model: item.dataset.clickModel, on: true });
    else if (event.target.closest("[data-click-toggle]")) change({ on: !pressed() });
  });

  // A job that ends may bring the chosen model in (a download), load it, or free the gpu lane for
  // its load: show the tool again, which prepares again while it's loading.
  document.addEventListener("job-ended", async () => {
    try {
      show(await (await fetch("/api/click-tool")).json());
    } catch {
      // The app didn't answer; the tool stays as it was.
    }
  });
}

const inBox = ([x1, y1, x2, y2], [x, y]) => x >= x1 && x <= x2 && y >= y1 && y <= y2;

// A point inside the shape: its outline's centre if that's inside it, else its box's.
function pointIn(shape) {
  const [bx, by, bw, bh] = shape.bbox;
  const centre = [bx + bw / 2, by + bh / 2];
  if (!shape.segments) return centre;
  const n = shape.segments.length;
  const middle = shape.segments.reduce(([sx, sy], [x, y]) => [sx + x / n, sy + y / n], [0, 0]);
  return contains(shape, middle) ? middle : centre;
}

// Is the point inside the shape: its polygon if it has one, else its box.
function contains(shape, [x, y]) {
  if (shape.segments) {
    let inside = false;
    const p = shape.segments;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
      const [xi, yi] = p[i];
      const [xj, yj] = p[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  const [bx, by, bw, bh] = shape.bbox;
  return inBox([bx, by, bx + bw, by + bh], [x, y]);
}
