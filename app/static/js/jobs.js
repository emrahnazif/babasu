// Jobs: keeps the drawer's job list and the top-bar count current, handles Cancel and Retry,
// and shows a toast for each job started while the page is open: it follows the job's progress
// and turns into "Finished" (or failed, cancelled…) when the job ends. The list itself is rendered by the server
// (partials/jobs.html); this file fetches it again and applies the changes.
//
// For other parts of the page: a "job-ended" event fires on document when a job ends (detail:
// the job's data-* values, e.g. projectId), and a "submitted" form (js/background-form.js) makes
// the list refresh at once, so a job it started shows up with its "Started" toast.
import { toast } from "./toast.js";

const panel = document.querySelector("#jobs-panel");
const count = document.querySelector("#jobs-count");

const ACTIVE_POLL = 1000; // while a job is queued or running
const IDLE_POLL = 5000; // otherwise, to notice jobs started elsewhere (e.g. by the agent)
const ENDED = { succeeded: "Finished", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted" };

// Apply a freshly rendered list to the current one, keeping the elements of jobs that only
// progressed: an element that stays the same (e.g. a progress bar) can animate to its new state.
function patchList(list, freshList) {
  list.dataset.active = freshList.dataset.active;
  const current = new Map([...list.querySelectorAll(":scope > .job")].map((job) => [job.dataset.id, job]));
  const items = [...freshList.children].map((item) => {
    const old = current.get(item.dataset.id);
    return old && old.dataset.status === item.dataset.status ? patchJob(old, item) : item;
  });
  // Moving nodes would restart their animations, so the list is only rebuilt when it changed.
  const sameOrder = items.length === list.children.length && items.every((item, i) => list.children[i] === item);
  if (!sameOrder) list.replaceChildren(...items);
}

function patchJob(old, fresh) {
  if (old.children.length !== fresh.children.length) return fresh;
  copyAttributes(fresh, old);
  [...fresh.children].forEach((child, i) => {
    const before = old.children[i];
    if (before.isEqualNode(child)) return;
    if (before.tagName === child.tagName && before.className === child.className && !child.children.length) {
      copyAttributes(child, before); // same kind of element: update it in place
      before.textContent = child.textContent;
    } else {
      before.replaceWith(child);
    }
  });
  return old;
}

function copyAttributes(from, to) {
  for (const { name, value } of from.attributes) to.setAttribute(name, value);
}

if (panel) {
  let known = new Map(); // job id -> status, as last seen
  const notes = new Map(); // job id -> the toast following it
  let lastHtml = "";
  let timer;

  // Compare the list with what was seen before: toast new and ended jobs, refresh regions.
  function update(announce) {
    const seen = new Map();
    for (const job of panel.querySelectorAll(".job")) {
      const { id, status, title, summary, message, progress, progressText } = job.dataset;
      seen.set(id, status);
      const before = known.get(id);
      const active = status === "queued" || status === "running";
      const started = `Started: ${title}`;
      if (announce && before === undefined && active) notes.set(id, toast(started));
      if (active) {
        // Queued, or running with no total yet: a bar without a known end.
        notes.get(id)?.set(started, [message, progressText].filter(Boolean).join(" · "),
          progress === "" ? null : Number(progress));
      } else if (announce && before !== status && ENDED[status]) {
        const note = notes.get(id);
        if (note?.open) note.set(`${ENDED[status]}: ${title}`, summary);
        else toast(`${ENDED[status]}: ${title}`, summary);
        notes.delete(id);
        document.dispatchEvent(new CustomEvent("job-ended", { detail: { ...job.dataset } }));
      }
    }
    known = seen;

    const active = Number(panel.querySelector(".job-list").dataset.active);
    count.textContent = active;
    count.hidden = active === 0;
    clearTimeout(timer);
    if (!document.hidden) timer = setTimeout(refresh, active ? ACTIVE_POLL : IDLE_POLL);
  }

  async function refresh() {
    try {
      const response = await fetch("/jobs/panel");
      const html = response.ok ? await response.text() : lastHtml;
      if (html !== lastHtml) {
        lastHtml = html;
        const fresh = new DOMParser().parseFromString(html, "text/html").querySelector(".job-list");
        patchList(panel.querySelector(".job-list"), fresh);
      }
    } catch {
      // Server busy or restarting: try again on the next tick.
    }
    update(true);
  }

  panel.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-job-action]");
    if (!button) return;
    button.disabled = true;
    const id = button.closest(".job").dataset.id;
    await fetch(`/api/jobs/${id}/${button.dataset.jobAction}`, { method: "POST" });
    refresh();
  });

  // A form that may have started a job (js/background-form.js): look now, not at the next tick.
  document.addEventListener("submitted", refresh);

  // No polling while the window is hidden; catch up as soon as it's visible again.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearTimeout(timer);
    else refresh();
  });

  update(false); // the page arrived with the current list: remember it, schedule the next check
}
