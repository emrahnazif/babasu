"""Code mode: Python the agent writes, for what no single API call does (going through many images,
filtering, combining projects), run in a child process: one per conversation, kept between runs,
so what one run leaves in variables the next can use (the agent works in steps: do, look, decide,
go on, on the same data). What it prints reaches the chat as it's printed.

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
import os
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
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


class _Printed(io.TextIOBase):
    """The child's stdout: each line printed goes to the app as it's printed (the chat shows it)."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._line = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._line += text
        if "\n" in self._line:
            done, self._line = self._line.rsplit("\n", 1)
            self._pipe.send(("print", done + "\n"))
        return len(text)

    def flush(self) -> None:
        if self._line:
            self._pipe.send(("print", self._line))
            self._line = ""


def _child(pipe, folder: str) -> None:
    """In the child: work in the conversation's folder (its files open by relative paths), and run
    each piece of code the app sends (a snippet, or a file of the folder) all in one namespace, so
    what one run leaves in variables and functions the next can use. Sends what's printed as it's
    printed, then ("done", error): None, or the error with the line it was on (its file, number
    and text)."""
    os.chdir(folder)
    printed = _Printed(pipe)
    shown = [-1]  # the last percentage sent in this run: a loop of any length sends at most ~100

    def progress(done: int, total: int) -> None:
        """Shows how far a long loop is, as a bar in the chat."""
        percent = int(100 * done / total) if total else 0
        if percent != shown[0]:
            shown[0] = percent
            pipe.send(("progress", done, total))

    api = Api(pipe)
    names = {"api": api, "Refused": Refused, "progress": progress, "__name__": "__main__"}
    while True:
        code, name = pipe.recv()  # name: CODE, or the file it was read from
        shown[0], error = -1, None
        label = "" if name == CODE else f"{name} "
        if name == CODE:
            names.pop("__file__", None)
        else:
            names["__file__"] = name
        try:
            with contextlib.redirect_stdout(printed):
                exec(compile(code, name, "exec"), names)
        except Exception as e:
            lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == name]
            where = f"{label}line {lines[-1]} ({code.splitlines()[lines[-1] - 1].strip()}): " if lines else ""
            error = where + "".join(traceback.format_exception_only(e)).strip()
        printed.flush()
        pipe.send(("done", error))


CODE = "<code>"  # what a snippet's tracebacks call it


class Session:
    """One conversation's code process: started at its first run, kept while it's used, so its
    variables last from one run to the next. A run that times out or crashes ends it."""

    def __init__(self, folder: Path) -> None:
        context = multiprocessing.get_context("spawn")
        self._pipe, there = context.Pipe()
        self._process = context.Process(target=_child, args=(there, str(folder)), daemon=True)
        self._process.start()
        there.close()  # the child's end is the child's: once it ends, reading here sees that at once
        self.used = time.monotonic()

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def close(self) -> None:
        if self._process.is_alive():
            self._process.kill()
        self._process.join()

    def run(self, code: str, call: Call, timeout: int = TIMEOUT, name: str = CODE) -> Iterator[dict]:
        """Run the code (read from the file `name`, if it was), making its API calls with `call` (here, in this thread), for at most
        `timeout` seconds. Yields what it shows as it comes, {"type": "output", "text"} (as much
        as the agent gets) and {"type": "progress", "done", "total"}; returns {output: what it
        printed (cut when long), error: why it stopped, or None}."""
        self.used = started = time.monotonic()
        self._pipe.send((code, name))
        kept, size = [], 0  # what's printed, up to OUTPUT_SIZE; how much in all
        while True:
            left = timeout - (time.monotonic() - started)
            if left <= 0 or not self._pipe.poll(left):
                self.close()
                return _result(kept, size, f"Stopped after {timeout} s; give it more time (seconds) or do less. {GONE}")
            try:
                message = self._pipe.recv()
            except EOFError:
                self.close()
                return _result(kept, size, f"The code's process ended without finishing (it crashed or was killed). {GONE}")
            if message[0] == "done":
                return _result(kept, size, message[1])
            if message[0] == "print":
                text = message[1][:max(OUTPUT_SIZE - size, 0)]
                size += len(message[1])
                if text:
                    kept.append(text)
                    yield {"type": "output", "text": text}
            elif message[0] == "progress":
                yield {"type": "progress", "done": message[1], "total": message[2]}
            else:
                _, method, path, body = message
                try:
                    self._pipe.send((True, call(method, path, body)))
                except Refused as e:
                    self._pipe.send((False, str(e)))


GONE = "Its variables are gone: the next run starts a new process."


def _result(kept: list[str], size: int, error: str | None) -> dict:
    text = "".join(kept)
    if size > OUTPUT_SIZE:
        text += f"\n… cut: {size:,} characters printed in all"
    return {"output": text, "error": error}


# Each conversation's code process, by its folder. One that hasn't run code for IDLE seconds is
# ended when the next run anywhere starts; at most SESSIONS live at once (the oldest goes).
IDLE = 30 * 60
SESSIONS = 4
_sessions: dict[Path, Session] = {}  # by the conversation's folder
_lock = threading.Lock()


def session(folder: Path) -> Session:
    """The code process working in a conversation's folder, started if it has none or its last
    one ended."""
    with _lock:
        now = time.monotonic()
        for other, found in list(_sessions.items()):
            if not found.alive or (other != folder and now - found.used > IDLE):
                found.close()
                del _sessions[other]
        if folder not in _sessions:
            while len(_sessions) >= SESSIONS:
                oldest = min(_sessions, key=lambda k: _sessions[k].used)
                _sessions.pop(oldest).close()
            _sessions[folder] = Session(folder)
        return _sessions[folder]


def close_all() -> None:
    """At shutdown (the server's lifespan): end every code process."""
    with _lock:
        for found in _sessions.values():
            found.close()
        _sessions.clear()
