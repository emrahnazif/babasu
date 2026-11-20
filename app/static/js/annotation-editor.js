// The annotation editor on an image page: MarkinJS (static/vendor) draws an image's annotations
// over it, in its classes' colours, and lets people change them. The page renders MarkinJS's
// structure (an <img> and an <svg> over it in a data-markin-stage) and the data, on the <svg>:
//   <svg viewBox="0 0 W H" data-annotations="[…]" data-hidden="[…]" data-hides="[names]"
//        data-classes="[{name, color}, …]" data-unnamed="#c82e2d" data-save="/api/…"
//        data-source="/api/…" data-project-id="3" data-frame="12" data-next-frame="15">
// data-annotations: what's drawn; data-hidden: the rest (the classes the project hides, data-hides,
// and on a video the other frames'), not drawn, saved back as they are. On a video (data-frame,
// the frame shown; data-next-frame, the one after it), js/video.js says when another frame is
// shown ("frame", detail {frame, next, playing}): the editor draws that frame's annotations, sets
// data-frame and data-next-frame (the click tool works on them), and dispatches "frame-shown"
// (detail {playing}). It saves the frame shown only (?frame=).
// Annotations are in the original image's pixels, the format babasu stores, except that MarkinJS
// calls an annotation's class `class_id` (any string): the class name goes in there on the way in,
// and comes back as `class` on the way out. When a job of the
// project ends (js/jobs.js's "job-ended", e.g. auto-annotating), the annotations are read again
// from data-source: what the job added shows up.
//
// Each completed edit saves the whole list, as it is when it's sent. One request at a time: edits
// made while one is on its way are sent together after it, so the last edit always wins. When
// they're through, the editor dispatches "annotations-saved" (detail: projectId) on document, for
// what the page shows about them (the toolbar's count is a js/refresh-region.js region listening to it).
//
// The class in the toolbar (#class-shown) is the selected object's, or with nothing selected the
// class new objects get; picking one in #class-menu changes both. A name typed there that isn't a
// class (New class, or Enter) becomes one; the project keeps it, with a colour of its own, once an
// object of it is saved, and the editor reads the classes back then. New objects come from the
// click tool (js/click-tool.js), which asks for a name when the project has no class yet.
import { clickTool } from "./click-tool.js";
import { toast } from "./toast.js";

const canvas = document.querySelector("svg[data-annotations]");

if (canvas) {
  const classes = JSON.parse(canvas.dataset.classes);
  let hides = new Set(JSON.parse(canvas.dataset.hides));
  let frame = canvas.dataset.frame ? Number(canvas.dataset.frame) : null; // a video's frame shown
  let rest = JSON.parse(canvas.dataset.hidden); // the annotations not drawn
  const editor = MarkinJS.create(canvas, {
    annotations: JSON.parse(canvas.dataset.annotations).map(toMarkin),
    colors: colors(classes, canvas.dataset.unnamed),
  });
  // What's drawn, as babasu stores it. MarkinJS's ids are its own handles; babasu doesn't keep them.
  const drawn = () =>
    editor.getAnnotations().map(({ id, class_id, class: _, ...a }) => ({ class: class_id, ...a, ...(frame !== null && { frame }) }));
  // Draw those of `annotations` that are shown (on the frame shown, of a class not hidden); keep the rest.
  const draw = (annotations) => {
    const shown = (a) => !hides.has(a.class.toLowerCase()) && (frame === null || (a.frame ?? 0) === frame);
    rest = annotations.filter((a) => !shown(a));
    editor.load(annotations.filter(shown).map(toMarkin));
  };
  // The image as the app has it now: its annotations, and its project's classes.
  const source = async () => {
    try {
      const response = await fetch(canvas.dataset.source);
      return response.ok ? await response.json() : null;
    } catch {
      return null; // the app didn't answer; the page shows what it had
    }
  };
  document.addEventListener("job-ended", async (event) => {
    if (event.detail.projectId !== canvas.dataset.projectId) return;
    const image = await source();
    if (!image) return;
    hides = new Set(image.project.classes.filter((c) => c.hidden).map((c) => c.name.toLowerCase()));
    draw(image.annotations);
  });
  document.addEventListener("frame", (event) => {
    const everything = [...drawn(), ...rest];
    const { next, playing } = event.detail;
    frame = event.detail.frame;
    canvas.dataset.frame = frame;
    if (next === null) delete canvas.dataset.nextFrame;
    else canvas.dataset.nextFrame = next;
    draw(everything);
    document.dispatchEvent(new CustomEvent("frame-shown", { detail: { playing } }));
  });
  // What's saved: everything, or on a video the frame shown (with its hidden classes' annotations).
  const saved = () => {
    const url = frame === null ? canvas.dataset.save : `${canvas.dataset.save}?frame=${frame}`;
    return { url, annotations: [...drawn(), ...rest.filter((a) => frame === null || (a.frame ?? 0) === frame)] };
  };
  editor.on("change", saver(saved));
  clickTool(editor, canvas, classTool(editor, classes, source));
}

function toMarkin({ class: name, ...annotation }) {
  return { ...annotation, class_id: name };
}

// MarkinJS's colours: each class's, and an object's with no class yet ("").
function colors(classes, unnamed) {
  return { "": unnamed, ...Object.fromEntries(classes.map((c) => [c.name, c.color])) };
}

// `saved()`: {url, annotations}, what to save where, as it is when it's sent.
function saver(saved) {
  let due = false; // an edit not sent yet
  let sending = false;
  return async function save() {
    due = true;
    if (sending) return;
    sending = true;
    while (due) {
      due = false;
      try {
        const { url, annotations } = saved();
        const body = JSON.stringify(annotations);
        const response = await fetch(url, { method: "PUT", headers: { "Content-Type": "application/json" }, body });
        if (!response.ok) {
          const { detail } = await response.json();
          throw new Error(typeof detail === "string" ? detail : response.statusText);
        }
      } catch (error) {
        toast("Couldn't save the annotations", error.message);
      }
    }
    sending = false;
    // What the page shows about them (the count) is read again.
    document.dispatchEvent(new CustomEvent("annotations-saved", { detail: { projectId: canvas.dataset.projectId } }));
  };
}

// Returns {current() (the class new objects get; null until the project has one), add(name) (a
// class typed, chosen)}.
function classTool(editor, classes, source) {
  const shown = document.querySelector("#class-shown");
  const menu = document.querySelector("#class-menu");
  const search = menu.querySelector(".menu__search");
  let newClass = classes[0]?.name ?? null; // the class new objects get

  const show = (name) => {
    const c = classes.find((c) => c.name === name);
    shown.textContent = name;
    shown.classList.toggle("plain", !c?.color);
    shown.style.setProperty("--badge-color", c?.color ?? "");
  };
  editor.on("select", ({ annotation }) => show(annotation.class_id));
  editor.on("deselect", () => newClass && show(newClass)); // a select follows if the selection moved

  function choose(name) {
    newClass = name;
    const selected = editor.getSelection();
    if (selected) editor.updateAnnotation(selected.id, { class_id: name });
    show(name);
    menu.hidePopover();
  }

  // A typed name: the class spelt that way, ignoring case, or a new one (no colour until the project keeps it).
  function typed(text) {
    const name = text.trim();
    if (!name) return;
    const known = classes.find((c) => c.name.toLowerCase() === name.toLowerCase());
    if (known) return choose(known.name);
    classes.push({ name, color: "" });
    const item = document.getElementById("class-item").content.firstElementChild.cloneNode(true);
    item.dataset.className = name;
    item.append(name);
    menu.querySelector("[data-new-class]").before(item);
    choose(name);
  }

  // Once saved, a new class has its colour: the classes are read back.
  document.addEventListener("annotations-saved", async () => {
    if (classes.every((c) => c.color)) return;
    const image = await source();
    if (!image) return;
    classes.splice(0, classes.length, ...image.project.classes.filter((c) => !c.hidden));
    editor.setOptions({ colors: colors(classes, canvas.dataset.unnamed) });
    for (const item of menu.querySelectorAll("[data-class-name]")) {
      const c = classes.find((c) => c.name === item.dataset.className);
      if (c) item.querySelector(".menu__swatch").style.setProperty("--swatch-color", c.color);
    }
    show(shown.textContent);
  });

  menu.addEventListener("click", (event) => {
    const item = event.target.closest("[data-class-name]");
    if (item) choose(item.dataset.className);
    else if (event.target.closest("[data-new-class]")) typed(search.value);
  });
  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault(); // or, the menu closed and focus back on its button, Enter opens it again
    typed(search.value);
  });
  return { current: () => newClass, add: typed };
}
