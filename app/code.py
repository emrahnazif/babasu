"""Code mode: Python the agent writes, for what no single API call does (going through many images,
filtering, combining projects), run in a child process.

The code's one way in is `api`, the app's API. Its calls go back over a pipe to the app, which
makes them in the agent's turn, so they're checked as the pages' are and recorded in History as
part of that turn. Nothing else is handed to the code (no database, no connection). It's the
person's own machine and their agent: Python can still open files, so "the API only" is how the
code is guided (app/docs/code.md), not a sandbox.

A child process of the app's own executable (multiprocessing, spawn), not `python` on the PATH: a
PyInstaller build has no interpreter, but runs its own children the same way (run.py calls
freeze_support). The child has the app's packages (pixelflow, numpy). Kept light: the child
imports this module, so it imports nothing of the app's at the top.
"""

import contextlib
import io
import multiprocessing
import time
import traceback
from collections.abc import Callable
from urllib.parse import urlencode

TIMEOUT = 60  # seconds a run may take unless the agent gives more
OUTPUT_SIZE = 8_000  # characters of what the code printed that come back; the rest is cut

# Makes one API call for the code: (method, path, body) -> its JSON answer; raises Refused.
Call = Callable[[str, str, object], object]


class Refused(Exception):
    """The API refused a call; the message is its reason."""


class Api:
    """The app's API, for code: api.get("/api/projects"), api.post(path, body), api.put, api.patch,
    api.delete. Each returns the JSON answer (None when there's none); a refusal raises Refused
    with the API's reason. api.images(project_id) goes through a project's images, page by page."""

    def __init__(self, pipe):
        self._pipe = pipe

    def _call(self, method: str, path: str, body=None):
        self._pipe.send(("call", method, path, body))
        ok, answer = self._pipe.recv()
        if not ok:
            raise Refused(answer)
        return answer

    def get(self, path: str):
        return self._call("GET", path)

    def post(self, path: str, body=None):
        return self._call("POST", path, body)

    def put(self, path: str, body=None):
        return self._call("PUT", path, body)

    def patch(self, path: str, body=None):
        return self._call("PATCH", path, body)

    def delete(self, path: str):
        return self._call("DELETE", path)

    def images(self, project_id: int, where: str = ""):
        """A project's images (those matching the filter `where`), each with its objects, read a
        page at a time. Don't change what a filter matches while going through it: the pages move."""
        page, pages = 1, 1
        while page <= pages:
            data = self.get(f"/api/projects/{project_id}?{urlencode({'page': page, 'where': where})}")
            pages = data["pages"]
            yield from data["images"]
            page += 1


def _child(code: str, pipe) -> None:
    """In the child: run the code with `api` and `Refused`, and send back what it printed and,
    if it failed, the error with the line of the code it was on (its number and text)."""
    printed = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(code, "<code>", "exec"), {"api": Api(pipe), "Refused": Refused, "__name__": "__main__"})
    except Exception as e:
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == "<code>"]
        where = f"line {lines[-1]} ({code.splitlines()[lines[-1] - 1].strip()}): " if lines else ""
        error = where + "".join(traceback.format_exception_only(e)).strip()
    pipe.send(("done", printed.getvalue(), error))


def run(code: str, call: Call, timeout: int = TIMEOUT) -> dict:
    """Run the code in a child process, making its API calls with `call` (here, in this thread),
    for at most `timeout` seconds. Returns {output: what it printed (cut when long), error: why it
    stopped, or None}."""
    context = multiprocessing.get_context("spawn")
    here, there = context.Pipe()
    process = context.Process(target=_child, args=(code, there), daemon=True)
    started = time.monotonic()
    process.start()
    there.close()  # the child's end is the child's: once it ends, reading here sees that at once
    try:
        while True:
            left = timeout - (time.monotonic() - started)
            if left <= 0 or not here.poll(left):
                return {"output": "", "error": f"Stopped after {timeout} s; give it more time (seconds) or do less."}
            try:
                message = here.recv()
            except EOFError:
                return {"output": "", "error": "The code's process ended without finishing (it crashed or was killed)."}
            if message[0] == "done":
                _, output, error = message
                if len(output) > OUTPUT_SIZE:
                    output = output[:OUTPUT_SIZE] + f"\n… cut: {len(output):,} characters printed in all"
                return {"output": output, "error": error}
            _, method, path, body = message
            try:
                here.send((True, call(method, path, body)))
            except Refused as e:
                here.send((False, str(e)))
    finally:
        if process.is_alive():
            process.kill()
        process.join()
