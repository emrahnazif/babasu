// Select (css/components/select.css; markup: the select() macro, partials/select.html).
// Picking an option puts its value in the hidden input and its text on the trigger, moves the
// check, and fires "change" on the input (it bubbles), as a <select> would. The menu closes
// itself (js/menu.js).
document.addEventListener("click", (event) => {
  const option = event.target.closest(".select__menu [role=option]");
  if (!option) return;
  const select = option.closest(".select");
  for (const other of select.querySelectorAll("[role=option]")) other.setAttribute("aria-selected", String(other === option));
  select.querySelector(".select__label").textContent = option.textContent.trim();
  const input = select.querySelector("input[type=hidden]");
  if (input.value === option.dataset.value) return;
  input.value = option.dataset.value;
  input.dispatchEvent(new Event("change", { bubbles: true }));
});

// Opening the list focuses the chosen option; closing it (a pick, Esc, a click outside) gives the
// focus back to the trigger when it was in the list. toggle doesn't bubble: listen on the way down.
document.addEventListener("toggle", (event) => {
  const menu = event.target;
  if (!menu.matches?.(".select__menu")) return;
  if (event.newState === "open") menu.querySelector("[aria-selected=true]")?.focus();
  else if (menu.contains(document.activeElement)) menu.closest(".select").querySelector(".select__trigger").focus();
}, true);
