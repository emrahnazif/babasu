// Toasts: short notices in the bottom-right corner, for any feature.
//   import { toast } from "./toast.js";  toast("Finished: Import images", "Added: 120");
// With a progress bar it stays until changed, and set() changes it (clicking dismisses it):
//   const note = toast("Started: Download", "", null);  // null: a bar with no known total yet
//   note.set("Started: Download", "52 MB / 694 MB", 0.08);  // 0 to 1
//   note.set("Finished: Download", "Done");  // no progress: the bar goes, it leaves on its own
// The markup is the #toast-template in base.html; the stack (#toasts) is a manual popover.
// The look and motion are in components/toast.css; here a toast only gets added, marked
// leaving (toast--leaving) and removed once its fade-out has ended.
const stack = document.querySelector("#toasts");
const template = document.querySelector("#toast-template");
const TOAST_TIME = 6000;

export function toast(title, text = "", progress = undefined) {
  const item = template.content.firstElementChild.cloneNode(true);
  const bar = item.querySelector(".toast__progress");
  let timer;

  const leave = () => {
    if (item.classList.contains("toast--leaving")) return;
    item.classList.add("toast--leaving");
    // Remove after the fade; the timeout covers reduced motion or a fade that never runs.
    const remove = () => {
      item.remove();
      if (!stack.children.length) stack.hidePopover();
    };
    item.addEventListener("transitionend", remove, { once: true });
    setTimeout(remove, 1000);
  };
  item.addEventListener("click", leave);

  function set(title, text = "", progress = undefined) {
    item.querySelector(".toast__title").textContent = title;
    item.querySelector(".toast__text").textContent = text;
    bar.hidden = progress === undefined;
    bar.classList.toggle("indeterminate", progress === null);
    if (typeof progress === "number") bar.style.setProperty("--value", progress);
    clearTimeout(timer);
    if (progress === undefined) timer = setTimeout(leave, TOAST_TIME);
  }
  set(title, text, progress);

  stack.append(item);
  if (!stack.matches(":popover-open")) stack.showPopover();
  // open: not dismissed or leaving, so set() still shows.
  return { set, get open() { return !item.classList.contains("toast--leaving"); } };
}

// The stack must stay above a dialog or drawer opened after it: re-open it (which moves it to
// the top of the top layer) whenever another popover opens. Toggle events don't bubble: capture.
document.addEventListener(
  "toggle",
  (event) => {
    if (event.newState === "open" && event.target !== stack && stack.matches(":popover-open")) {
      stack.hidePopover();
      stack.showPopover();
    }
  },
  true,
);
