// The annotation editor on an image page: MarkinJS (static/vendor) draws an image's annotations
// over it, in its classes' colours, and lets people change them. The page renders MarkinJS's
// structure (an <img> and an <svg> over it in a data-markin-stage) and the data, on the <svg>:
//   <svg viewBox="0 0 W H" data-annotations="[…]" data-classes="[{id, name, color}, …]"
//        data-save="/api/…" data-source="/api/…" data-project-id="3">
// Annotations are in the original image's pixels, the format babasu stores. Without data-save
// (a frozen project, or one without classes) the editor only shows them. When a job of the
// project ends (js/jobs.js's "job-ended", e.g. auto-annotating), the annotations are read again
// from data-source: what the job added shows up.
//
// Each completed edit saves the whole list. One request at a time: edits made while one is on
// its way are sent together after it, so the last edit always wins. When they're through, the
// editor dispatches "annotations-saved" (detail: projectId) on document, for what the page shows
// about them (the toolbar's count is a js/refresh-region.js region listening to it).
//
// The class in the toolbar (#class-shown) is the selected object's, or with nothing selected the
// class new objects get; picking one in #class-menu changes both. New objects come from the click
// tool (js/click-tool.js).
import { clickTool } from "./click-tool.js";
import { toast } from "./toast.js";

const canvas = document.querySelector("svg[data-annotations]");

if (canvas) {
  const classes = JSON.parse(canvas.dataset.classes);
  const editor = MarkinJS.create(canvas, {
    annotations: JSON.parse(canvas.dataset.annotations),
    colors: Object.fromEntries(classes.map((c) => [c.id, c.color])),
  });
  document.addEventListener("job-ended", async (event) => {
    if (event.detail.projectId !== canvas.dataset.projectId) return;
    try {
      const response = await fetch(canvas.dataset.source);
      if (response.ok) editor.load((await response.json()).annotations);
    } catch {
      // The app didn't answer; the page shows what it had.
    }
  });
  const url = canvas.dataset.save;
  if (url) {
    editor.on("change", saver(url));
    clickTool(editor, canvas, classTool(editor, classes));
  } else {
    editor.disable();
  }
}

function saver(url) {
  let waiting = null; // the newest list not sent yet
  let sending = false;
  return async function save(annotations) {
    waiting = annotations;
    if (sending) return;
    sending = true;
    while (waiting) {
      // MarkinJS's ids are its own handles on the annotations; babasu doesn't store them.
      const body = JSON.stringify(waiting.map(({ id, ...annotation }) => annotation));
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

// Returns a function giving the class new objects get.
function classTool(editor, classes) {
  const shown = document.querySelector("#class-shown");
  let newClass = classes[0].id; // the class new objects get

  const show = (classId) => {
    const c = classes.find((c) => c.id === classId);
    shown.textContent = c.name;
    shown.style.setProperty("--badge-color", c.color);
  };
  editor.on("select", ({ annotation }) => show(annotation.class_id));
  editor.on("deselect", () => show(newClass)); // a select follows if the selection moved

  document.querySelector("#class-menu").addEventListener("click", (event) => {
    const item = event.target.closest("[data-class-id]");
    if (!item) return;
    newClass = Number(item.dataset.classId);
    const selected = editor.getSelection();
    if (selected) editor.updateAnnotation(selected.id, { class_id: newClass });
    show(newClass);
  });
  return () => newClass;
}
