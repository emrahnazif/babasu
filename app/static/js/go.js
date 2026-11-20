// Back and forward through the pages seen in this window, as a browser's buttons do (the desktop
// window has none): <button data-go="-1"> and <button data-go="1">, next to the sidebar toggle.
// Where the browser says whether there's somewhere to go (the Navigation API), the button that
// has nowhere is disabled.
const buttons = document.querySelectorAll("[data-go]");

for (const button of buttons) button.addEventListener("click", () => history.go(Number(button.dataset.go)));

function update() {
  if (!window.navigation) return; // unknown: both stay enabled
  for (const button of buttons) button.disabled = button.dataset.go < 0 ? !navigation.canGoBack : !navigation.canGoForward;
}

update();
// This page again, from the browser's cache: WebKit says where one can go only after pageshow, by
// navigatesuccess (sent for a move within the page too); pageshow is for browsers that don't send
// it then.
window.addEventListener("pageshow", update);
window.navigation?.addEventListener("navigatesuccess", update);
