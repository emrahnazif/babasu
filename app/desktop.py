"""Opens the app in a native window via pywebview."""

import webview

from app import config
from app.server import start_server


def main() -> None:
    server, url = start_server()

    webview.create_window(
        config.APP_NAME,
        url,
        width=config.WINDOW_SIZE[0],
        height=config.WINDOW_SIZE[1],
        min_size=config.WINDOW_MIN_SIZE,
    )
    webview.start()  # blocks until the window is closed

    server.should_exit = True
