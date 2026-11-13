// The class rows of the project form (projects/form.html):
//   <tbody id="classes" data-palette='["#e11d48", …]'> rows </tbody>
//   <template id="class-row"> a new row </template>  <button data-add-class>  <button data-remove-class>
// Add class appends a new row with the first palette colour no row uses (a removed row's too).
// Remove takes a new row away; a saved class's row (data-saved) stays, its fields disabled so it
// isn't sent, with a note saying its annotations are hidden, and Remove becomes Undo.
const rows = document.getElementById("classes");
const template = document.getElementById("class-row");

function addRow() {
  const row = template.content.firstElementChild.cloneNode(true);
  const palette = JSON.parse(rows.dataset.palette);
  const used = new Set([...rows.querySelectorAll("input[type=color]")].map((input) => input.value.toLowerCase()));
  row.querySelector("input[type=color]").value = palette.find((color) => !used.has(color)) ?? palette[0];
  rows.append(row);
  row.querySelector("input[name=class_name]").focus();
}

function toggleRemoved(row, button) {
  const removed = button.textContent === "Remove";
  for (const input of row.querySelectorAll("input")) input.disabled = removed;
  for (const note of row.querySelectorAll("td.muted > span")) note.hidden = !note.hidden;
  button.textContent = removed ? "Undo" : "Remove";
}

document.addEventListener("click", (event) => {
  if (event.target.closest("[data-add-class]")) addRow();
  const remove = event.target.closest("[data-remove-class]");
  if (!remove) return;
  const row = remove.closest("tr");
  if (row.hasAttribute("data-saved")) toggleRemoved(row, remove);
  else row.remove();
});
