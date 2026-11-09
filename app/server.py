"""FastAPI app, plus a helper that runs it with uvicorn in a background thread."""

import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import config
from app.routes import pages

app = FastAPI(title=config.APP_NAME, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
app.include_router(pages.router)


def start_server() -> tuple[uvicorn.Server, str]:
    """Start uvicorn on a free localhost port. Returns the server and its URL."""
    server = uvicorn.Server(uvicorn.Config(app, host=config.HOST, port=0))  # 0 = OS picks a free port
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn failed to start")
        time.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    return server, f"http://{config.HOST}:{port}/"
