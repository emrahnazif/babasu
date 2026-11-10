"""Opens the app in a native window via pywebview."""

import json

import webview
from webview.dom import DOMEventHandler

from app import config
from app.server import start_server


class DesktopApi:
    """Native features the page can call as window.pywebview.api.<method>() (desktop window only).
    Every public method here is callable from the page, so keep it to what the UI needs."""

    def choose_files(self, accept: list[str]) -> list[str]:
        """Open the native file dialog for several files; `accept` is suffixes like [".jpg", ".png"].
        Returns the chosen paths (empty if cancelled)."""
        file_types = (f"Accepted files ({';'.join('*' + s for s in accept)})",) if accept else ()
        chosen = webview.windows[0].create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True, file_types=file_types)
        return list(chosen or [])


def _connect_drop_zones(window: webview.Window) -> None:
    """On every page load: elements marked data-drop-paths get the real paths of the files and
    folders dropped on them, as a "paths" event (detail: list of paths). pywebview only gives
    full paths to listeners registered from Python, hence this bridge (see js/drop-zone.js)."""
    for element in window.dom.get_elements("[data-drop-paths]"):

        def on_drop(event: dict, element_id: str = element.id) -> None:
            files = event.get("dataTransfer", {}).get("files", [])
            paths = [f["pywebviewFullPath"] for f in files if "pywebviewFullPath" in f]
            detail = json.dumps(paths)
            window.run_js(
                f"document.getElementById({json.dumps(element_id)})"
                f"?.dispatchEvent(new CustomEvent('paths', {{ detail: {detail} }}))"
            )

        element.on("drop", DOMEventHandler(on_drop, prevent_default=True))


def main() -> None:
    server, server_thread, url = start_server()

    window = webview.create_window(
        config.APP_NAME,
        url,
        width=config.WINDOW_SIZE[0],
        height=config.WINDOW_SIZE[1],
        min_size=config.WINDOW_MIN_SIZE,
        js_api=DesktopApi(),
    )
    window.events.loaded += _connect_drop_zones
    webview.start()  # blocks until the window is closed

    # Let the server shut down fully: its lifespan stops running jobs (up to 5 s).
    server.should_exit = True
    server_thread.join(timeout=10)
