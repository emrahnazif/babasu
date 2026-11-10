"""FastAPI app, plus a helper that runs it with uvicorn in a background thread."""

import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import config, db, jobs
from app.routes import jobs as jobs_routes
from app.routes import pages, projects, settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Owns startup and shutdown of everything long-lived (database schema, job workers)."""
    db.migrate()
    jobs.run_workers()
    yield
    jobs.shutdown()


app = FastAPI(title=config.APP_NAME, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
# check_dir=False: the folder appears with the first stored file.
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR, check_dir=False), name="media")
app.include_router(pages.router)
app.include_router(projects.router)
app.include_router(settings.router)
app.include_router(jobs_routes.router)


def start_server() -> tuple[uvicorn.Server, threading.Thread, str]:
    """Start uvicorn on a free localhost port. Returns the server, its thread and its URL.

    To stop: set `server.should_exit` and join the thread, so the lifespan's shutdown
    (which stops running jobs) completes before the process exits.
    """
    # port=0: the OS picks a free port. timeout_graceful_shutdown: open requests (e.g. a streaming
    # agent reply) get a few seconds to end before they're cancelled, so closing never hangs.
    server = uvicorn.Server(uvicorn.Config(app, host=config.HOST, port=0, timeout_graceful_shutdown=3))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn failed to start")
        time.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://{config.HOST}:{port}/"
