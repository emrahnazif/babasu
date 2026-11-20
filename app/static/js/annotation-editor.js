// The annotation editor on an image page: MarkinJS (static/vendor) draws an image's annotations
// over it, in its classes' colours, and lets people change them. The page renders MarkinJS's
// structure (an <img> and an <svg> over it in a data-markin-stage) and the data, on the <svg>:
//   <svg viewBox="0 0 W H" data-annotations="[…]" data-hidden="[…]" data-classes="[{name, color}, …]"
//        data-unnamed="#c82e2d" data-save="/api/…" data-source="/api/…" data-project-id="3">
// data-hidden: the annotations of the classes the project hides; not drawn, saved back as they are.
// Annotations are in the original image's pixels, the format babasu stores, except that MarkinJS
// calls an annotation's class `class_id` (any string): the class name goes in there on the way in,
// and comes back as `class` on the way out. When a job of the
// project ends (js/jobs.js's "job-ended", e.g. auto-annotating), the annotations are read again
// from data-source: what the job added shows up.
//
// Each completed edit saves the whole list. One request at a time: edits made while one is on
// its way are sent together after it, so the last edit always wins. When they're through, the
// editor dispatches "annotations-saved" (detail: projectId) on document, for what the page shows
// about them (the toolbar's count is a js/refresh-region.js region listening to it).
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
  let hidden = JSON.parse(canvas.dataset.hidden);
  const editor = MarkinJS.create(canvas, {
    annotations: JSON.parse(canvas.dataset.annotations).map(toMarkin),
    colors: colors(classes, canvas.dataset.unnamed),
  });
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
    const hides = new Set(image.project.classes.filter((c) => c.hidden).map((c) => c.name.toLowerCase()));
    hidden = image.annotations.filter((a) => hides.has(a.class.toLowerCase()));
    editor.load(image.annotations.filter((a) => !hides.has(a.class.toLowerCase())).map(toMarkin));
  });
  editor.on("change", saver(canvas.dataset.save, () => hidden));
  clickTool(editor, canvas, classTool(editor, classes, source));
}

function toMarkin({ class: name, ...annotation }) {
  return { ...annotation, class_id: name };
}

// MarkinJS's colours: each class's, and an object's with no class yet ("").
function colors(classes, unnamed) {
  return { "": unnamed, ...Object.fromEntries(classes.map((c) => [c.name, c.color])) };
}

// `kept`: the annotations saved with what the editor has (the hidden classes').
function saver(url, kept) {
  let waiting = null; // the newest list not sent yet
  let sending = false;
  return async function save(annotations) {
    waiting = annotations;
    if (sending) return;
    sending = true;
    while (waiting) {
      // MarkinJS's ids are its own handles on the annotations; babasu doesn't store them.
      const drawn = waiting.map(({ id, class_id, class: _, ...annotation }) => ({ class: class_id, ...annotation }));
      const body = JSON.stringify([...drawn, ...kept()]);
      waiting = null;
      try {
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
