// Menu (css/components/menu.css).
// Picking an item (a link or button) closes the menu, as in any menu. This runs during the
// click, before the item's own action (e.g. popovertarget opening a panel), so what it opens
// isn't nested in the menu and stays open on its own. Labels (e.g. a checkbox) leave it open.
document.addEventListener("click", (event) => {
  event.target.closest("a.menu__item, button.menu__item")?.closest(".menu:popover-open")?.hidePopover();
});

// A menu's search field narrows its items to those whose text has what's typed. Elements with
// data-search-echo show the text; items with data-search-typed show only while there's some.
document.addEventListener("input", (event) => {
  const field = event.target.closest(".menu__search");
  if (!field) return;
  const menu = field.closest(".menu");
  const text = field.value.trim();
  for (const item of menu.querySelectorAll(".menu__item")) {
    item.hidden = item.hasAttribute("data-search-typed")
      ? !text
      : Boolean(text) && !item.textContent.toLowerCase().includes(text.toLowerCase());
  }
  for (const echo of menu.querySelectorAll("[data-search-echo]")) echo.textContent = text;
});

// ↑/↓ move between a menu's items (from its search field too), Home/End jump to the first or
// last; Enter or Space picks (they're links and buttons), Esc closes (the popover). Items of a
// menu nested in it (e.g. a select in a panel) are that menu's own.
const MOVES = ["ArrowDown", "ArrowUp", "Home", "End"];
document.addEventListener("keydown", (event) => {
  if (!MOVES.includes(event.key)) return;
  const menu = event.target.closest?.(".menu:popover-open");
  if (!menu) return;
  const field = event.target.matches("input, textarea, select");
  if (field && !(event.target.matches(".menu__search") && event.key.startsWith("Arrow"))) return;
  const items = [...menu.querySelectorAll("a.menu__item, button.menu__item")]
    .filter((item) => !item.hidden && !item.disabled && item.closest(".menu") === menu);
  if (!items.length) return;
  event.preventDefault();
  const at = items.indexOf(document.activeElement);
  const next = { ArrowDown: at + 1, ArrowUp: at < 0 ? -1 : at - 1, Home: 0, End: -1 }[event.key];
  items.at(next % items.length).focus();
});
