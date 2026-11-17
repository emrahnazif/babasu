// Fill: a button that puts its text into a field elsewhere on the page, the cursor at its end, for
// the person to finish (e.g. Home's "Find all…" into the chat's field). It knows nothing about the
// field: a "reveal" event asks whatever hides it (a collapsed sidebar) to show it.
//   <button type="button" data-fill="chat-input" value="Find all ">Find all…</button>
for (const button of document.querySelectorAll("[data-fill]")) {
  button.addEventListener("click", () => {
    const field = document.getElementById(button.dataset.fill);
    if (!field) return;
    field.dispatchEvent(new CustomEvent("reveal", { bubbles: true }));
    field.value = button.value;
    field.focus();
    field.setSelectionRange(field.value.length, field.value.length);
  });
}
