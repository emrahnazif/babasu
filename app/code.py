"""Code mode: Python the agent writes, for what no single API call does (going through many images,
filtering, combining projects), run in a child process: one per conversation, kept between runs,
so what one run leaves in variables the next can use (the agent works in steps: do, look, decide,
go on, on the same data). What it prints reaches the chat as it's printed.

The code changes things only through `api`, the app's API. Its calls go back over a pipe to the
app, which makes them in the agent's turn, so they're checked as the pages' are and recorded in
History as part of that turn. Reading is open: `babasu` (app/workbench.py) gives the projects and
their images in vision terms, each image with the path of its file (the app says where it is, over
the same pipe), so the code reads pixels with pixelflow as any vision code does. No database or
connection is handed to it. It's the person's own machine and their agent: "change through the API"
is how the code is guided (app/docs/code.md), not a sandbox.

A child process of the app's own executable (multiprocessing, spawn), not `python` on the PATH: a
PyInstaller build has no interpreter, but runs its own children the same way (run.py calls
freeze_support). The child has the app's packages (pixelflow, numpy). Kept light: the child
imports this module, so it imports nothing of the app's at the top.
"""

import contextlib
import io
import multiprocessing
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path

TIMEOUT = 60  # seconds a run may take unless the agent gives more
STOP_CHECK = 0.1  # seconds between looks at whether the agent's turn was stopped (the code, the model)
OUTPUT_SIZE = 8_000  # characters of what the code printed that come back; the rest is cut

# Makes one API call for the code: (method, path, body) -> its JSON answer; raises Refused.
Call = Callable[[str, str, object], object]
# Where an image's file is, for the code: media_id -> {path, video}; raises Refused (no such image).
Locate = Callable[[int], dict]


class Refused(Exception):
    """The API refused a call; the message is its reason."""


class Api:
    """The app's API, for code: api.get("/api/projects"), api.post(path, body), api.put, api.patch,
    api.delete. Each returns the JSON answer (None when there's none); a refusal raises Refused
    with the API's reason."""

    def __init__(self, pipe):
        self._pipe = pipe

    def _call(self, method: str, path: str, body=None):
        return self._ask(("call", method, path, body))

    def _file(self, media_id: int) -> dict:
        """Where an image's file is: {path, video} (babasu's Image.file)."""
        return self._ask(("file", media_id))

    def _ask(self, message: tuple):
        self._pipe.send(message)
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


class _Printed(io.TextIOBase):
    """The child's stdout: each line printed goes to the app as it's printed (the chat shows it)."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._line = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._line += text
        if "\n" in text:
            done, self._line = self._line.rsplit("\n", 1)
            self._pipe.send(("print", done + "\n"))
        return len(text)

    def flush(self) -> None:
        if self._line:
            self._pipe.send(("print", self._line))
            self._line = ""


IDLE = 30 * 60  # seconds a conversation's process waits for its next run, then ends (freeing what it holds)


def _child(pipe, folder: str, tools: frozenset[str]) -> None:
    """In the child: work in the conversation's folder (its files open by relative paths), and run
    each piece of code the app sends all in one namespace, so
    what one run leaves in variables and functions the next can use. Sends what's printed as it's
    printed, then ("done", error): None, or the error with the line it was on (its file, number
    and text; `tools`: the agent's tool names, to say they aren't in code). Ends after IDLE seconds without a run."""
    os.chdir(folder)
    printed = _Printed(pipe)
    shown = -1  # the last percentage sent in this run: a loop of any length sends at most ~100

    def progress(done: int, total: int) -> None:
        """Shows how far a long loop is, as a bar in the chat."""
        nonlocal shown
        percent = int(100 * done / total) if total else 0
        if percent != shown:
            shown = percent
            pipe.send(("progress", done, total))

    from app import workbench  # here, not at the top: it imports this module

    api = Api(pipe)
    workbench.connect(api, progress)
    sys.modules["babasu"] = workbench  # `import babasu` is the one the code has
    names = {"api": api, "babasu": workbench, "Refused": Refused, "progress": progress, "__name__": "__main__"}
    while pipe.poll(IDLE):
        try:
            code = pipe.recv()
        except EOFError:  # the app ended (closed, or crashed): so does this
            return
        name, shown, error = "<code>", -1, None
        try:
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):  # warnings too
                exec(compile(code, name, "exec"), names)
        except SystemExit as e:  # sys.exit() ends the run, not the process
            error = None if e.code in (None, 0) else f"It exited with {e.code}."
        except Exception as e:
            lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == name]
            where = ""
            if lines:
                where = f"line {lines[-1]} ({code.splitlines()[lines[-1] - 1].strip()}): "
            error = where + "".join(traceback.format_exception_only(e)).strip() + _hint(e, tools)
        printed.flush()
        pipe.send(("done", error))


def _hint(e: Exception, tools: frozenset[str]) -> str:
    """How to correct what code written as if it were elsewhere gets wrong: importing `api`, calling a tool."""
    if isinstance(e, ModuleNotFoundError) and e.name == "api":
        return ". `api` is there already, without an import: api.get(path), api.post(path, body)…"
    if isinstance(e, NameError) and e.name in tools:
        return (f". The tools aren't in code: api.get(path) reads, api.post, put, patch and delete write; "
                f"call {e.name} itself outside the code")
    return ""


class Session:
    """One conversation's code process: started at its first run, kept while it's used, so its
    variables last from one run to the next. A run that times out or crashes ends it, and it ends
    itself after IDLE seconds without a run."""

    def __init__(self, folder: Path, tools: frozenset[str]) -> None:
        context = multiprocessing.get_context("spawn")
        self._pipe, there = context.Pipe()
        self._process = context.Process(target=_child, args=(there, str(folder), tools), daemon=True)
        self._process.start()
        there.close()  # the child's end is the child's: once it ends, reading here sees that at once
        self.used = time.monotonic()
        self.running = False  # a run is under way: the process isn't ended to make room for another

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def close(self) -> None:
        self._process.kill()  # nothing if it has ended
        self._process.join()

    def _ended(self, why: str) -> str:
        """End the process after a run that didn't finish; why, for the agent."""
        self.close()
        return f"{why} Its variables are gone: the next run starts a new process."

    def run(self, code: str, call: Call, locate: Locate, stop: threading.Event, timeout: int = TIMEOUT) -> Iterator[dict]:
        """Run the code, making its API calls with `call` and saying where its images' files are
        with `locate` (here, in this thread), for at most `timeout` seconds, or until `stop` is set. Yields what it shows as it
        comes, {"type": "output", "text"} (as much as the agent gets) and {"type": "progress",
        "done", "total"}; returns {output: what it printed (cut when long), error: why it stopped,
        or None}. A run left before the code is done (the app failed to answer one of its calls,
        or the turn ended) ends the process: the code would wait for an answer that never comes."""
        self.running = True
        try:
            return (yield from self._run(code, call, locate, stop, timeout))
        except BaseException:
            self.close()
            raise
        finally:
            self.running = False

    def _run(self, code: str, call: Call, locate: Locate, stop: threading.Event, timeout: int) -> Iterator[dict]:
        started = time.monotonic()
        self._pipe.send(code)
        kept, size = [], 0  # what's printed, up to OUTPUT_SIZE; how much in all
        while True:
            left = timeout - (time.monotonic() - started)
            if left <= 0:
                error = self._ended(f"Stopped after {timeout} s; give it more time (seconds) or do less.")
                break
            if stop.is_set():
                error = self._ended("Stopped by the person.")
                break
            if not self._pipe.poll(min(left, STOP_CHECK)):
                continue
            try:
                message = self._pipe.recv()
            except EOFError:
                error = self._ended("The code's process ended without finishing (it crashed or was killed).")
                break
            if message[0] == "done":
                error = message[1]
                break
            if message[0] == "print":
                text = message[1][:max(OUTPUT_SIZE - size, 0)]
                size += len(message[1])
                if text:
                    kept.append(text)
                    yield {"type": "output", "text": text}
            elif message[0] == "progress":
                yield {"type": "progress", "done": message[1], "total": message[2]}
            else:  # a question it waits on: an API call, or where an image's file is
                try:
                    answer = call(*message[1:]) if message[0] == "call" else locate(message[1])
                    self._pipe.send((True, answer))
                except Refused as e:
                    self._pipe.send((False, str(e)))
        cut = f"\n… cut: {size:,} characters printed in all" if size > OUTPUT_SIZE else ""
        return {"output": "".join(kept) + cut, "error": error}


SESSIONS = 4  # conversation processes alive at once: starting another ends the least recently used idle one
_sessions: dict[Path, Session] = {}  # by the conversation's folder
_lock = threading.Lock()


def session(folder: Path, tools: frozenset[str]) -> Session:
    """The code process working in a conversation's folder, started if it has none or its last
    one ended; `tools`, the agent's tool names (code calling one is told how to correct it)."""
    with _lock:
        for other, found in list(_sessions.items()):
            if not found.alive:
                found.close()
                del _sessions[other]
        if folder not in _sessions:
            if len(_sessions) >= SESSIONS and (idle := [k for k, s in _sessions.items() if not s.running]):
                _sessions.pop(min(idle, key=lambda k: _sessions[k].used)).close()  # one running in another chat goes on
            _sessions[folder] = Session(folder, tools)
        _sessions[folder].used = time.monotonic()  # the most recently used now, before its run starts
        return _sessions[folder]


def close_all() -> None:
    """At shutdown (the server's lifespan): end every code process."""
    with _lock:
        for found in _sessions.values():
            found.close()
        _sessions.clear()
