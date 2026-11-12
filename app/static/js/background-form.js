// Background form: a form sent without leaving the page, with the button that sent it.
//   <form method="post" action="…" data-background-form>
// On success the form is reset (data-background-form="keep": left as typed), the popover around
// it closes, and a "submitted" event bubbles from the form (js/jobs.js refreshes its list). On
// failure a toast gives the server's reason. Without JS the form simply posts as usual.
import { toast } from "./toast.js";

// The server's reason for refusing, if it gave one in words (FastAPI's {"detail": "…"}).
async function reason(response) {
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") return detail;
  } catch {
    // Not JSON: the status says it.
  }
  return `The server answered ${response.status}.`;
}

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("form[data-background-form]");
  if (!form) return;
  event.preventDefault();
  let response = null;
  try {
    response = await fetch(form.action, { method: "POST", body: new FormData(form, event.submitter) });
  } catch {
    // The server didn't answer; reported below.
  }
  if (!response?.ok) {
    toast("Couldn't send the form", response ? await reason(response) : "The app didn't answer.");
    return;
  }
  if (form.dataset.backgroundForm !== "keep") form.reset();
  form.closest("[popover]")?.hidePopover();
  form.dispatchEvent(new CustomEvent("submitted", { bubbles: true }));
});
