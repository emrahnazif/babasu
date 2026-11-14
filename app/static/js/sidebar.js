// Sidebar: drag its right border to resize, use the top-bar button to collapse/expand.
// Changes are saved to settings, so the server renders the same state on the next page.

const app = document.querySelector(".app");
const resizer = document.querySelector(".app__resizer");
const toggle = document.querySelector("#sidebar-toggle");

function saveSidebar(change) {
  fetch("/settings/ui", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(change),
  });
}

if (resizer) {
  const min = Number(resizer.getAttribute("aria-valuemin"));
  const max = Number(resizer.getAttribute("aria-valuemax"));
  let width = Number(resizer.getAttribute("aria-valuenow"));
  let saveTimer;

  const setWidth = (value) => {
    width = Math.round(Math.min(max, Math.max(min, value)));
    app.style.setProperty("--sidebar-width", `${width}px`);
    resizer.setAttribute("aria-valuenow", width);
  };

  resizer.addEventListener("pointerdown", (event) => {
    event.preventDefault(); // no text selection while dragging
    resizer.setPointerCapture(event.pointerId);
    const start = width;
    const left = app.getBoundingClientRect().left;
    const onMove = (e) => setWidth(e.clientX - left);
    resizer.addEventListener("pointermove", onMove);
    // Fires on pointerup and on an interrupted drag alike.
    resizer.addEventListener("lostpointercapture", () => {
      resizer.removeEventListener("pointermove", onMove);
      if (width !== start) saveSidebar({ sidebar_width: width });
    }, { once: true });
  });

  resizer.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -16, ArrowRight: 16 }[event.key];
    if (!step) return;
    event.preventDefault();
    setWidth(width + step);
    clearTimeout(saveTimer); // save once the key is released, not on every repeat
    saveTimer = setTimeout(() => saveSidebar({ sidebar_width: width }), 300);
  });
}

toggle?.addEventListener("click", () => {
  const collapsed = app.classList.toggle("app--collapsed");
  toggle.setAttribute("aria-expanded", String(!collapsed));
  saveSidebar({ sidebar_collapsed: collapsed });
});
