// Undo and Redo for the whole app, as in an editor: ⌘Z undoes the latest action (a change made on
// a page, a job, an agent's turn: POST /api/history/undo), ⇧⌘Z does again what the latest undo
// undid (/api/history/redo). Not while typing in a field (that's the field's own undo), nor on a
// page with an undo of its own: an element marked data-own-undo (e.g. the annotation editor).
// The page reloads to show the change, then a toast says what was undone, with Redo (Undo after a
// redo). It's kept for that one load in sessionStorage; where that can't be, the page stays and
// the toast shows at once.
import { reason } from "./background-form.js";
import { matches, parse, typing } from "./shortcuts.js";
import { toast } from "./toast.js";

const KEY = "undone"; // the toast to show after the reload
const UNDO = parse("mod+z");
const REDO = parse("mod+shift+z");

function show({ redo, text }) {
  toast(redo ? "Redone" : "Undone", text, undefined, { label: redo ? "Undo" : "Redo", run: () => step(!redo) });
}

async function step(redo) {
  const response = await fetch(`/api/history/${redo ? "redo" : "undo"}`, { method: "POST" }).catch(() => null);
  if (!response?.ok) {
    toast(redo ? "Couldn't redo" : "Couldn't undo", response ? await reason(response) : "The app didn't answer.");
    return;
  }
  const done = { redo, text: (await response.json()).changed || "Nothing to put back" };
  try {
    sessionStorage.setItem(KEY, JSON.stringify(done));
  } catch {
    return show(done); // can't keep it over a reload: say it here
  }
  location.reload();
}

try {
  const done = sessionStorage.getItem(KEY);
  sessionStorage.removeItem(KEY);
  if (done) show(JSON.parse(done));
} catch {} // no storage: nothing was kept

// On window, so it runs after listeners on the page; a key one of them has handled is theirs.
window.addEventListener("keydown", (event) => {
  if (event.repeat || event.defaultPrevented || !(matches(UNDO, event) || matches(REDO, event))) return;
  if (typing(event.target) || document.querySelector("[data-own-undo]")) return;
  event.preventDefault();
  step(event.shiftKey);
});
