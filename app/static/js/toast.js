// Toasts: short notices in the bottom-right corner, for any feature.
//   import { toast } from "./toast.js";  toast("Finished: Import images", "Added: 120");
// The markup is the #toast-template in base.html; the stack (#toasts) is a manual popover.
// The look and motion are in components/toast.css; here a toast only gets added, marked
// leaving (toast--leaving) and removed once its fade-out has ended.
const stack = document.querySelector("#toasts");
const template = document.querySelector("#toast-template");
const TOAST_TIME = 6000;

export function toast(title, text = "") {
  const item = template.content.firstElementChild.cloneNode(true);
  item.querySelector(".toast__title").textContent = title;
  item.querySelector(".toast__text").textContent = text;

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
  setTimeout(leave, TOAST_TIME);

  stack.append(item);
  if (!stack.matches(":popover-open")) stack.showPopover();
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
