// The Test button in the model dialogs (Settings → Models): tests the connection with what's in
// the form, before it's saved, and shows the result in the dialog's [data-test-result].
document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-test-model]");
  if (!button) return;
  const form = button.closest("form");
  const result = form.querySelector("[data-test-result]");
  const fields = Object.fromEntries(new FormData(form));
  button.disabled = true;
  result.className = "muted";
  result.textContent = "Testing…";
  try {
    const response = await fetch("/api/models/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields),
    });
    const outcome = response.ok ? await response.json() : { ok: false, message: `The app answered ${response.status}.` };
    result.className = outcome.ok ? "muted" : "error";
    result.textContent = outcome.message;
  } catch {
    result.className = "error";
    result.textContent = "The app didn't answer.";
  } finally {
    button.disabled = false;
  }
});
