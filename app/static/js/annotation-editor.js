// The annotation editor on an image page: MarkinJS (static/vendor) draws an image's annotations
// over it, in its classes' colours, and lets people change them. The page renders the data:
//   <img src="…" data-annotations="[…]" data-classes="[{id, name, color}, …]" data-save="/api/…">
// Annotations are in the original image's pixels, the format babasu stores. Without data-save
// (a frozen project, or one without classes) the editor only shows them.
//
// Each completed edit saves the whole list. One request at a time: edits made while one is on
// its way are sent together after it, so the last edit always wins.
//
// The class in the toolbar (#class-shown) is the selected object's, or with nothing selected the
// class new objects get; picking one in #class-menu changes both.
import { toast } from "./toast.js";

const image = document.querySelector("img[data-annotations]");

if (image) {
  const classes = JSON.parse(image.dataset.classes);
  const editor = MarkinJS.create(image, {
    annotations: JSON.parse(image.dataset.annotations),
    colors: Object.fromEntries(classes.map((c) => [c.id, c.color])),
  });
  const url = image.dataset.save;
  if (url) {
    editor.on("change", saver(url));
    classTool(editor, classes);
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
  };
}

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
}
