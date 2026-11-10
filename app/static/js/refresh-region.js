// Refresh region: part of a page reloaded from the server when something happens elsewhere.
//   <div id="…" data-refresh-on="job-ended" data-refresh-if="projectId=3">
// When an event named in data-refresh-on fires on document, and its detail matches
// data-refresh-if ("key=value", optional), the region is replaced by the element with the same id
// in a fresh copy of this page. Several events at once cause one reload.
const names = new Set(
  [...document.querySelectorAll("[data-refresh-on]")].flatMap((region) => region.dataset.refreshOn.split(" ")),
);
const due = new Set(); // regions waiting for the reload

function matches(region, detail) {
  const condition = region.dataset.refreshIf;
  if (!condition) return true;
  const [key, value] = condition.split("=");
  return String(detail?.[key]) === value;
}

async function reload() {
  const regions = [...due];
  due.clear();
  const page = new DOMParser().parseFromString(await (await fetch(location.href)).text(), "text/html");
  for (const region of regions) {
    const fresh = page.getElementById(region.id);
    if (fresh) region.replaceWith(fresh);
  }
}

for (const name of names) {
  document.addEventListener(name, (event) => {
    const regions = document.querySelectorAll(`[data-refresh-on~="${name}"]`);
    const wasIdle = due.size === 0;
    for (const region of regions) if (matches(region, event.detail)) due.add(region);
    if (wasIdle && due.size) queueMicrotask(reload); // after the events of this moment are in
  });
}
