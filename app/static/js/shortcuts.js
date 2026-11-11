// Keyboard shortcuts for the whole app. A shortcut is another way to click something, so the
// element declares it and this file clicks it:
//   <button popovertarget="chat-history" data-shortcut="mod+k" title="Search chats">
// Keys: "mod" (⌘ on macOS, Ctrl elsewhere), "shift", "alt", then the key (as event.key, e.g. "k",
// "arrowleft"), joined with "+".
// Rules:
//   - with mod, a shortcut works even while typing (e.g. ⌘K from the chat box); without it,
//     not while typing in a field
//   - a disabled element doesn't respond; one that isn't on this page simply isn't there
//   - don't declare the system's own (⌘C/V/X/Z, ⌘W, ⌘Q…)
// The shortcut is also added to the element's tooltip and aria-keyshortcuts.
const mac = /Mac|iPhone|iPad/.test(navigator.platform);
const SYMBOLS = mac ? { mod: "⌘", shift: "⇧", alt: "⌥" } : { mod: "Ctrl+", shift: "Shift+", alt: "Alt+" };
const ARIA = { mod: mac ? "Meta" : "Control", shift: "Shift", alt: "Alt" };
// Keys whose name isn't what a person reads (event.key, lower case -> shown, ARIA name).
const KEYS = { arrowleft: ["←", "ArrowLeft"], arrowright: ["→", "ArrowRight"], arrowup: ["↑", "ArrowUp"], arrowdown: ["↓", "ArrowDown"] };

function parse(shortcut) {
  const parts = shortcut.toLowerCase().split("+");
  const key = parts.pop();
  return { key, mod: parts.includes("mod"), shift: parts.includes("shift"), alt: parts.includes("alt") };
}

function matches({ key, mod, shift, alt }, event) {
  return (
    event.key.toLowerCase() === key &&
    (mac ? event.metaKey : event.ctrlKey) === mod &&
    (mac ? event.ctrlKey : event.metaKey) === false &&
    event.shiftKey === shift &&
    event.altKey === alt
  );
}

const typing = (target) => target.closest?.("input, textarea, select, [contenteditable]") != null;

const elements = [...document.querySelectorAll("[data-shortcut]")].map((element) => {
  const shortcut = parse(element.dataset.shortcut);
  const modifiers = ["mod", "shift", "alt"].filter((m) => shortcut[m]);
  const [keyShown, keyAria] = KEYS[shortcut.key] ?? [shortcut.key.toUpperCase(), shortcut.key.toUpperCase()];
  const shown = modifiers.map((m) => SYMBOLS[m]).join("") + keyShown;
  if (element.title) element.title += ` (${shown})`;
  element.setAttribute("aria-keyshortcuts", [...modifiers.map((m) => ARIA[m]), keyAria].join("+"));
  return { element, shortcut };
});

// On window, so it runs after listeners on the page (e.g. the annotation editor's, where ← moves
// the selected box); a key one of them has handled is theirs.
window.addEventListener("keydown", (event) => {
  if (event.repeat || event.isComposing || event.defaultPrevented) return;
  const found = elements.find(({ shortcut }) => matches(shortcut, event));
  if (!found || found.element.disabled) return;
  if (!found.shortcut.mod && typing(event.target)) return;
  event.preventDefault();
  found.element.click();
});
