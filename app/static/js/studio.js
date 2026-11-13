// Studio: run a workflow from a view. Components (app/view_components.py):
//   <div class="dropzone" data-studio-drop data-run='{"workflow_id": …, "prompts": […]}'>   drop or pick images
//   <button data-studio-run data-run='{"workflow_id": …, "project_id": …, "count": …}'>       run on a project's images
// A drop zone stores the images (POST /api/media), then starts the run; a button starts it as
// given (POST /api/runs). The run is a job like any other: the jobs list (js/jobs.js) shows its
// progress, and when it ends ("job-ended") the page is shown again with its results. A drop zone
// with an input named "prompts" sends what was typed in it (comma-separated) as the prompts.
import { reason } from "./background-form.js";
import { toast } from "./toast.js";

async function post(url, init) {
  const response = await fetch(url, { method: "POST", ...init });
  if (!response.ok) throw new Error(await reason(response));
  return response.json();
}

function say(element, text) {
  const status = element.querySelector("[data-status]") || element.parentElement.querySelector("[data-status]");
  if (status) status.textContent = text;
}

async function run(element, body) {
  const { job_id } = await post("/api/runs", { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  say(element, "Running: the jobs list shows how far it is.");
  document.addEventListener("job-ended", function ended(event) {
    if (event.detail.id !== String(job_id)) return;
    document.removeEventListener("job-ended", ended);
    if (event.detail.status === "succeeded") location.reload();
    else say(element, "");  // the jobs list says why
  });
  element.dispatchEvent(new CustomEvent("submitted", { bubbles: true }));  // the jobs list looks now
}

async function dropped(zone, files) {
  const images = [...files].filter((file) => file.type.startsWith("image/"));
  if (!images.length) return;
  say(zone, `Storing ${images.length} image${images.length === 1 ? "" : "s"}…`);
  try {
    const form = new FormData();
    for (const image of images) form.append("files", image);
    const stored = await post("/api/media", { body: form });
    const body = { ...JSON.parse(zone.dataset.run), media_ids: stored.map((m) => m.media_id) };
    const typed = zone.querySelector("input[name=prompts]");
    if (typed) body.prompts = typed.value.split(",").map((p) => p.trim()).filter(Boolean);
    await run(zone, body);
  } catch (error) {
    say(zone, "");
    toast("Couldn't start the run", error.message);
  }
}

for (const zone of document.querySelectorAll("[data-studio-drop]")) {
  const picker = zone.querySelector("input[type=file]");
  zone.addEventListener("click", (event) => {
    if (!event.target.closest("input:not([type=file]), label")) picker.click();
  });
  zone.addEventListener("keydown", (event) => {
    if (event.target === zone && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); picker.click(); }
  });
  picker.addEventListener("change", () => dropped(zone, picker.files));
  zone.addEventListener("dragover", (event) => { event.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("over");
    dropped(zone, event.dataTransfer.files);
  });
}

for (const button of document.querySelectorAll("[data-studio-run]")) {
  button.addEventListener("click", () => run(button, JSON.parse(button.dataset.run)).catch((error) => {
    toast("Couldn't start the run", error.message);
  }));
}
