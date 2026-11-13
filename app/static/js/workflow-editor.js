// The workflow builder (workflows/edit.html): mozo's editor mounted into [data-workflow-editor],
// driven from our bar. The editor keeps the graph; the bar keeps the name and description
// ([data-about]), puts them into the document, and saves it: PUT to data-save-to, or, for a new
// workflow or a built-in's copy (no data-save-to), POST, then the page moves to the saved one.
// Test runs the workflow on a chosen image; every node's output shows on the canvas.
import { mount } from "/mozo/workflow/assets/editor.js";
import { reason } from "./background-form.js";
import { toast } from "./toast.js";

const element = document.querySelector("[data-workflow-editor]");
const about = document.querySelectorAll("[data-about]");
const unsaved = document.querySelector("[data-unsaved]");
const fileInput = document.querySelector("[data-test-file]");
const chooseButton = document.querySelector("[data-test-choose]");
const testButton = document.querySelector("[data-test]");
const saveButton = document.querySelector("[data-save]");

const editor = mount(element, { document: JSON.parse(element.dataset.document), toolbar: false });
let image = null;  // the image Test runs on

function changed(yes) {
  unsaved.hidden = !yes;
}

editor.on("change", () => changed(true));
about.forEach((input) => input.addEventListener("input", () => changed(true)));

// Leaving with unsaved changes asks first.
window.addEventListener("beforeunload", (event) => {
  if (!unsaved.hidden) event.preventDefault();
});

function documentToSave() {
  const saved = editor.getDocument();
  about.forEach((input) => { saved[input.name] = input.value.trim(); });
  return saved;
}

saveButton.addEventListener("click", async () => {
  const to = element.dataset.saveTo;
  saveButton.disabled = true;
  let response = null;
  try {
    response = await fetch(to || "/api/workflows", {
      method: to ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(documentToSave()),
    });
  } catch {
    // The server didn't answer; reported below.
  }
  saveButton.disabled = false;
  if (!response?.ok) {
    toast("Couldn't save the workflow", response ? await reason(response) : "The app didn't answer.");
    return;
  }
  const saved = await response.json();
  changed(false);
  if (to) toast("Saved", saved.name);
  else location.assign(`/workflows/${saved.id}/edit`);
});

fileInput.addEventListener("change", () => {
  image = fileInput.files[0] || image;
  if (image) chooseButton.querySelector("span").textContent = image.name;
  fileInput.value = "";
});

// Ask for an image; resolves to the one Test now runs on (still none if the person cancelled).
function choose() {
  return new Promise((resolve) => {
    fileInput.addEventListener("change", () => resolve(image), { once: true });
    fileInput.addEventListener("cancel", () => resolve(image), { once: true });
    fileInput.click();
  });
}

chooseButton.addEventListener("click", choose);

testButton.addEventListener("click", async () => {
  if (!image && !(await choose())) return;
  testButton.disabled = true;
  await editor.test(image);  // the editor shows each node's output, or why it failed
  testButton.disabled = false;
});
