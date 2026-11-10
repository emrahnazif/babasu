// Drop zone: drag files or folders onto it, or click it to choose files. It knows nothing about
// what the paths are for: it collects them as hidden form fields named by data-name, so a form
// receives them like any other field.
//   <div class="dropzone" id="…" data-drop-paths data-name="paths" data-accept=".jpg,.png"
//        role="button" tabindex="0"><p>Drop files or folders here</p><ul></ul></div>
// Real paths only exist in the desktop window: app/desktop.py sends them as a "paths" event.
// A browser hides them, so there the zone says so and a typed path is the way in.
const desktop = () => Boolean(window.pywebview?.api);

function setUp(zone) {
  const list = zone.querySelector("ul");
  const paths = new Set();

  function add(newPaths) {
    for (const path of newPaths) {
      if (paths.has(path)) continue;
      paths.add(path);
      const input = Object.assign(document.createElement("input"), { type: "hidden", name: zone.dataset.name, value: path });
      const item = Object.assign(document.createElement("li"), { textContent: path, title: path });
      zone.append(input);
      list.append(item);
    }
  }

  function clear() {
    paths.clear();
    list.replaceChildren();
    for (const input of zone.querySelectorAll("input[type=hidden]")) input.remove();
  }

  // The browser must be told not to open a dropped file itself.
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("over");
    });
  }
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("over");
    if (!desktop()) zone.querySelector("p").textContent = "Dropping works in the desktop app. Enter a path instead.";
  });
  zone.addEventListener("paths", (event) => add(event.detail));

  async function choose() {
    if (!desktop()) return;
    const accept = zone.dataset.accept ? zone.dataset.accept.split(",") : [];
    add(await window.pywebview.api.choose_files(accept));
  }
  zone.addEventListener("click", choose);
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  });
  zone.closest("form")?.addEventListener("reset", clear);
}

for (const zone of document.querySelectorAll("[data-drop-paths]")) setUp(zone);
