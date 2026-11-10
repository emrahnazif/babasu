// Background form: a form sent without leaving the page.
//   <form method="post" action="…" data-background-form>
// On success the form is reset, the popover around it (e.g. a dialog) closes, and a "submitted"
// event bubbles from the form for whoever cares (e.g. js/jobs.js refreshes its list). On failure a
// toast says so. Without JS the form simply posts as usual.
import { toast } from "./toast.js";

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("form[data-background-form]");
  if (!form) return;
  event.preventDefault();
  let response = null;
  try {
    response = await fetch(form.action, { method: "POST", body: new FormData(form) });
  } catch {
    // The server didn't answer; reported below.
  }
  if (!response?.ok) {
    toast("Couldn't send the form", response ? `The server answered ${response.status}.` : "The app didn't answer.");
    return;
  }
  form.reset();
  form.closest("[popover]")?.hidePopover();
  form.dispatchEvent(new CustomEvent("submitted", { bubbles: true }));
});
