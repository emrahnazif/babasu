// Drop zone: drag files or folders onto it, or click it to choose files. It knows nothing about
// what the paths are for: it collects them as hidden form fields named by data-name, so a form
// receives them like any other field: the form it's in, or the one data-form names (by id) when
// it's elsewhere on the page. That form's reset empties it, as its Clear button does.
// With data-preview (a URL answering HTML for the paths posted to it), it shows what they hold.
// Its actions (Clear, and a submit button sending the form elsewhere) show once it holds something.
//   <div class="dropzone" id="…" data-drop-paths data-name="paths" data-preview="/files/preview"
//        data-accept=".jpg,.png"><button type="button">Drop files or folders here</button>
//     <output></output><div class="actions"><button class="button secondary" type="button" data-clear>Clear</button></div></div>
// Real paths only exist in the desktop window: app/desktop.py sends them as a "paths" event.
// A browser hides them, so there the zone says so and a typed path is the way in.
const desktop = () => Boolean(window.pywebview?.api);

function setUp(zone) {
  const output = zone.querySelector("output");
  const paths = new Set();
  let asked = 0; // previews asked for: one answered after a later one was asked is dropped

  async function show() {
    const mine = ++asked;
    output.textContent = "Looking at them…";
    const body = new URLSearchParams([...paths].map((path) => ["paths", path]));
    const response = await fetch(zone.dataset.preview, { method: "POST", body });
    if (mine === asked) output.innerHTML = response.ok ? await response.text() : "Couldn't look at them.";
  }

  function add(newPaths) {
    const fresh = newPaths.filter((path) => !paths.has(path));
    for (const path of fresh) {
      paths.add(path);
      const input = Object.assign(document.createElement("input"), { type: "hidden", name: zone.dataset.name, value: path });
      if (zone.dataset.form) input.setAttribute("form", zone.dataset.form);
      zone.append(input);
    }
    if (fresh.length && zone.dataset.preview) show();
  }

  function clear() {
    paths.clear();
    asked++; // a preview on its way is no longer wanted
    output.replaceChildren();
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
    if (!desktop()) zone.querySelector("button").textContent = "Dropping works in the desktop app. Enter a path instead.";
  });
  zone.addEventListener("paths", (event) => add(event.detail));

  async function choose() {
    if (!desktop()) return;
    const accept = zone.dataset.accept ? zone.dataset.accept.split(",") : [];
    add(await window.pywebview.api.choose_files(accept));
  }
  // A click anywhere chooses files (its text is the button, for the keyboard), but on what it shows
  // and its actions.
  zone.addEventListener("click", (event) => {
    if (event.target.closest("[data-clear]")) clear();
    else if (!event.target.closest("output, .actions")) choose();
  });
  (zone.dataset.form ? document.getElementById(zone.dataset.form) : zone.closest("form"))?.addEventListener("reset", clear);
}

for (const zone of document.querySelectorAll("[data-drop-paths]")) setUp(zone);
