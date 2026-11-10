// Command dialog: a popover with a search field over results from the server (css: command.css).
//   <div class="command" popover data-command-src="/url">
//     <input class="command__input"> … <div class="command__list"></div>
// Opening it (e.g. with a popovertarget button) loads src?q= into the list, and
// typing reloads it. Up/Down move through the .command__item buttons, Enter picks one (clicks it).
for (const command of document.querySelectorAll(".command")) {
  const input = command.querySelector(".command__input");
  const list = command.querySelector(".command__list");
  let timer;
  let latest = 0; // only the newest search may fill the list

  const items = () => [...list.querySelectorAll(".command__item")];
  const selected = () => list.querySelector('.command__item[aria-selected="true"]');

  function select(item) {
    selected()?.removeAttribute("aria-selected");
    if (!item) return;
    item.setAttribute("aria-selected", "true");
    item.scrollIntoView({ block: "nearest" });
  }

  async function load() {
    const id = ++latest;
    const response = await fetch(`${command.dataset.commandSrc}?q=${encodeURIComponent(input.value.trim())}`);
    const html = await response.text();
    if (id !== latest) return;
    list.innerHTML = html;
    select(items()[0]);
  }

  command.addEventListener("toggle", (event) => {
    if (event.newState !== "open") return;
    input.value = "";
    load();
    input.focus();
  });

  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(load, 150);
  });

  input.addEventListener("keydown", (event) => {
    const all = items();
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!all.length) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      select(all[(all.indexOf(selected()) + step + all.length) % all.length]);
    } else if (event.key === "Enter" && !event.isComposing) {
      event.preventDefault();
      selected()?.click();
    }
  });

  // The pointer selects too, so the highlight is always the one Enter or a click picks.
  list.addEventListener("pointermove", (event) => {
    const item = event.target.closest(".command__item");
    if (item && item !== selected()) select(item);
  });
}
