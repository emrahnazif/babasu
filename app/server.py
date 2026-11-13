"""FastAPI app, plus a helper that runs it with uvicorn in a background thread."""

import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from mozo.workflow.api import workflow_router

from app import agent, config, db, jobs, views, vision
from app.filters import FilterError
from app.routes import agent as agent_routes
from app.routes import agents as agents_routes
from app.routes import jobs as jobs_routes
from app.routes import pages, projects, settings
from app.routes import runs as runs_routes
from app.routes import views as views_routes
from app.routes import vision as vision_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Owns startup and shutdown of everything long-lived (database schema, job workers, loaded models)."""
    db.migrate()
    vision.setup()
    agent.recover()
    jobs.run_workers()
    yield
    agent.stop_all()
    jobs.shutdown()
    vision.unload_all()


app = FastAPI(title=config.APP_NAME, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
# check_dir=False: the folder appears with the first stored file.
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR, check_dir=False), name="media")
app.include_router(pages.router)
app.include_router(projects.router)
app.include_router(settings.router)
app.include_router(jobs_routes.router)
app.include_router(agent_routes.router)
app.include_router(agents_routes.router)
app.include_router(vision_routes.router)
app.include_router(runs_routes.router)
app.include_router(views_routes.router)
# mozo's workflow editor and the routes it calls (the builder's canvas, /mozo/workflow): its Test
# loads models through ours, and whole runs are our jobs, so it offers no Run of its own.
app.include_router(workflow_router(models=vision.load, process=False), prefix="/mozo")
app.add_exception_handler(FilterError, projects.filter_error)
app.add_exception_handler(views.ViewError, views_routes.view_error)
views.setup(app)


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
