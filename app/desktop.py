"""Opens the app in a native window via pywebview."""

import json
import sys

import webview
from webview.dom import DOMEventHandler

from app import config
from app.server import start_server
from app.templating import templates


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


# Where the window can be dragged from in the unified title bar: the bars marked data-window-drag,
# except their links, buttons and fields (those stay clickable) and popovers opened from them.
DRAG_REGIONS = (
    "[data-window-drag], "
    "[data-window-drag] :not(a, button, input, select, textarea, label, [popover],"
    " a *, button *, label *, [popover] *)"
)


def _unified_title_bar(window: webview.Window) -> None:
    """macOS: no separate title bar. The page runs to the top edge and its top bars hold the
    traffic lights, as in most Mac apps. An empty toolbar gives the bar the standard height with
    the traffic lights centred in it; macOS keeps them placed on resize and in full screen.
    The page is told through the `unified_title_bar` template flag (room for the traffic lights),
    and the bars become drag regions; Windows and Linux keep their usual title bar."""
    import AppKit
    from PyObjCTools import AppHelper

    templates.env.globals["unified_title_bar"] = True
    webview.settings["DRAG_REGION_SELECTOR"] = DRAG_REGIONS
    webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True  # a click on a button isn't a drag

    def apply() -> None:
        native = window.native
        frame = native.frame()  # full-size content would shrink the window by the old bar's height
        native.setStyleMask_(native.styleMask() | AppKit.NSWindowStyleMaskFullSizeContentView)
        native.setTitlebarAppearsTransparent_(True)
        native.setTitleVisibility_(AppKit.NSWindowTitleHidden)
        # pywebview paints the title bar view with a flat colour (webview/platforms/cocoa.py, "Set the
        # titlebar color"); clear it, finding the view the same way, so the page shows through.
        native.contentView().superview().subviews().lastObject().setBackgroundColor_(None)
        native.setToolbar_(AppKit.NSToolbar.alloc().initWithIdentifier_("babasu"))
        native.setToolbarStyle_(AppKit.NSWindowToolbarStyleUnified)
        native.setFrame_display_(frame, False)

    # AppKit calls belong on the main thread.
    window.events.before_show += lambda: AppHelper.callAfter(apply)


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
    if sys.platform == "darwin":
        _unified_title_bar(window)
    webview.start()  # blocks until the window is closed

    # Let the server shut down fully: its lifespan stops running jobs (up to 5 s).
    server.should_exit = True
    server_thread.join(timeout=10)
