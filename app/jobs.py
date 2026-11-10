"""Background jobs: long work (imports, training, ...) with progress, logs, cancel and retry.

A job kind is a function `f(ctx, **params) -> dict | None`, registered with `@kind(name, lane)`.
Each lane runs one job at a time, oldest first, in its own thread, so an import never waits
for a training run and two trainings never share the GPU.

The database is the only channel between a job and the app: progress goes into the job row,
and a stop request (`stop_reason`: 'user' or 'shutdown') comes back from it. A job therefore
behaves the same in a thread or, later, in a subprocess.

Crash safety (product.md "Jobs"):
- A job must leave data consistent if it stops at any moment (commit small, complete steps)
  and must be safe to run again (e.g. an import skips images it already has).
- Closing the app asks running jobs to stop; they end as "interrupted".
- On startup, jobs still marked queued or running (the app crashed or was killed) become
  "interrupted". Nothing restarts by itself; Retry starts a new job with the same parameters.
"""

import json
import sqlite3
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass

from app import config, db

LANES = ("io", "gpu")
LOG_DIR = config.DATA_DIR / "jobs"
SYNC_INTERVAL = 0.25  # seconds between progress writes (each one also reads the stop request)
IDLE_WAKE = 10.0  # an idle lane rechecks the queue this often, besides being woken by start()
ACTIVE = ("queued", "running")
RETRYABLE = ("failed", "cancelled", "interrupted")


@dataclass(frozen=True)
class Kind:
    func: Callable[..., dict | None]
    lane: str
    title: str  # shown in the jobs drawer, e.g. "Import images"


_kinds: dict[str, Kind] = {}
_wake = {lane: threading.Event() for lane in LANES}
_stopping = threading.Event()  # ends the worker loops; jobs themselves are stopped through the DB
_threads: list[threading.Thread] = []


def kind(name: str, lane: str, title: str):
    """Register a job function under `name`, to run in `lane`."""
    assert lane in LANES, lane

    def register(func):
        _kinds[name] = Kind(func, lane, title)
        return func

    return register


class Context:
    """What a running job uses to report progress, write its log and notice a stop request."""

    def __init__(self, conn: sqlite3.Connection, job_id: int, log):
        self._conn, self._id, self._log = conn, job_id, log
        self.done, self.total_count, self.message = 0, None, ""
        self.stop_reason: str | None = None  # 'user' or 'shutdown', once the job is asked to stop
        self._last_sync = 0.0

    def total(self, n: int) -> None:
        self.total_count = n
        self._sync(force=True)

    def advance(self, n: int = 1, message: str | None = None) -> None:
        self.done += n
        if message is not None:
            self.message = message
        self._sync(force=self.done == self.total_count)

    def log(self, line: str) -> None:
        self._log.write(line + "\n")
        self._log.flush()  # a crash keeps everything logged so far

    @property
    def cancelled(self) -> bool:
        """True once the job should stop. Check it between items, and return when it's set."""
        self._sync()
        return self.stop_reason is not None

    def _sync(self, force: bool = False) -> None:
        """Write progress and read the stop request, in one statement, at most every SYNC_INTERVAL."""
        now = time.monotonic()
        if force or now - self._last_sync >= SYNC_INTERVAL:
            self._last_sync = now
            with self._conn:
                self.stop_reason = self._conn.execute(
                    "UPDATE job SET done = ?, total = ?, message = ? WHERE id = ? RETURNING stop_reason",
                    (self.done, self.total_count, self.message, self._id),
                ).fetchone()[0]


# --- Starting and controlling jobs (any thread, any connection) ---


def start(conn: sqlite3.Connection, kind_name: str, params: dict, project_id: int | None = None) -> int:
    """Queue a job and return its id. Commits, so the lane's worker can see it."""
    lane = _kinds[kind_name].lane
    with conn:
        job_id = conn.execute(
            "INSERT INTO job (kind, project_id, params) VALUES (?, ?, ?)",
            (kind_name, project_id, json.dumps(params)),
        ).lastrowid
    _wake[lane].set()
    return job_id


def cancel(conn: sqlite3.Connection, job_id: int) -> None:
    """Stop a job: a queued one at once, a running one at its next check."""
    with conn:
        conn.execute(
            f"UPDATE job SET status = 'cancelled', finished_at = {db.NOW} WHERE id = ? AND status = 'queued'",
            (job_id,),
        )
        conn.execute("UPDATE job SET stop_reason = 'user' WHERE id = ? AND status = 'running'", (job_id,))


def retry(conn: sqlite3.Connection, job_id: int) -> int | None:
    """Start a new job with the same kind and parameters. None if the job can't be retried."""
    row = conn.execute("SELECT kind, params, project_id, status FROM job WHERE id = ?", (job_id,)).fetchone()
    if row is None or row["status"] not in RETRYABLE or row["kind"] not in _kinds:  # same rule as can_retry
        return None
    return start(conn, row["kind"], json.loads(row["params"]), row["project_id"])


# --- Reading ---


_SELECT = "SELECT job.*, project.name AS project_name FROM job LEFT JOIN project ON project.id = job.project_id"


def _summary(result: dict | None, error: str | None) -> str:
    """One line for people: the error, or the result (a list counts as its length)."""
    if error:
        return error
    parts = [f"{key.replace('_', ' ').capitalize()}: {len(v) if isinstance(v, list) else v}"
             for key, v in (result or {}).items()]
    return " · ".join(parts)


def _job(row: sqlite3.Row) -> dict:
    known = _kinds.get(row["kind"])
    result = json.loads(row["result"]) if row["result"] else None
    return dict(row) | {
        "title": known.title if known else row["kind"],
        "params": json.loads(row["params"]),
        "result": result,
        "summary": _summary(result, row["error"]),
        "active": row["status"] in ACTIVE,
        "can_cancel": row["status"] in ACTIVE and not row["stop_reason"],
        "can_retry": row["status"] in RETRYABLE and known is not None,
    }


def list_recent(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Active jobs first, then the most recent finished ones."""
    rows = conn.execute(
        f"{_SELECT} ORDER BY job.status IN ('queued', 'running') DESC, job.id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_job(row) for row in rows]


def overview(conn: sqlite3.Connection) -> dict:
    """What the jobs drawer shows: recent jobs, and how many are active (for the top-bar count)."""
    recent = list_recent(conn)
    return {"jobs": recent, "active": sum(job["active"] for job in recent)}


def get(conn: sqlite3.Connection, job_id: int) -> dict | None:
    row = conn.execute(f"{_SELECT} WHERE job.id = ?", (job_id,)).fetchone()
    return _job(row) if row else None


def read_log(job_id: int, offset: int | None = 0, limit: int = 65536) -> tuple[str, int]:
    """Log text from byte `offset` (at most `limit` bytes), and the offset to continue from.

    offset=None reads the last `limit` bytes (the tail).
    """
    path = LOG_DIR / f"{job_id}.log"
    if not path.exists():
        return "", offset or 0
    if offset is None:
        offset = max(0, path.stat().st_size - limit)
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read(limit)
    return data.decode("utf-8", errors="replace"), offset + len(data)


# --- Lifecycle: called by the server's lifespan, at start and at shutdown ---


def run_workers() -> None:
    """Recover from a previous crash, then start one worker thread per lane."""
    LOG_DIR.mkdir(exist_ok=True)
    with closing(db.connect()) as conn:
        _interrupt(
            conn,
            running="The app stopped before this job finished.",
            queued="The app stopped before this job started.",
        )
    _stopping.clear()
    for lane in LANES:
        thread = threading.Thread(target=_worker, args=(lane,), name=f"jobs-{lane}", daemon=True)
        thread.start()
        _threads.append(thread)


def shutdown(timeout: float = 5.0) -> None:
    """Ask running jobs to stop and wait up to `timeout` seconds for them.

    Jobs still queued, and any that don't stop in time, are marked interrupted; a thread that
    is still running dies with the process.
    """
    with closing(db.connect()) as conn:
        with conn:
            conn.execute("UPDATE job SET stop_reason = 'shutdown' WHERE status = 'running'")
        _stopping.set()
        for event in _wake.values():
            event.set()
        deadline = time.monotonic() + timeout
        for thread in _threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        _threads.clear()
        _interrupt(
            conn,
            running="The app was closed while this job was running.",
            queued="The app was closed before this job started.",
        )


def _interrupt(conn: sqlite3.Connection, running: str, queued: str) -> None:
    """Mark every queued or running job interrupted, with the matching reason."""
    with conn:
        conn.execute(
            f"""UPDATE job SET status = 'interrupted', finished_at = {db.NOW},
                   error = CASE status WHEN 'running' THEN ? ELSE ? END
                WHERE status IN ('queued', 'running')""",
            (running, queued),
        )


def _worker(lane: str) -> None:
    while not _stopping.is_set():
        job = _claim_next([name for name, k in _kinds.items() if k.lane == lane])
        if job is None:
            _wake[lane].wait(timeout=IDLE_WAKE)
            _wake[lane].clear()
            continue
        _run(job)


def _claim_next(kinds: list[str]) -> sqlite3.Row | None:
    """Mark the oldest queued job of these kinds as running, and return it."""
    if not kinds:
        return None
    marks = ", ".join("?" * len(kinds))
    with closing(db.connect()) as conn, conn:
        return conn.execute(
            f"""UPDATE job SET status = 'running', started_at = {db.NOW}
                WHERE id = (SELECT id FROM job WHERE status = 'queued' AND kind IN ({marks}) ORDER BY id LIMIT 1)
                RETURNING *""",
            kinds,
        ).fetchone()


def _run(job: sqlite3.Row) -> None:
    with closing(db.connect()) as conn, (LOG_DIR / f"{job['id']}.log").open("a", encoding="utf-8") as log:
        ctx = Context(conn, job["id"], log)
        result, error = None, None
        try:
            result = _kinds[job["kind"]].func(ctx, **json.loads(job["params"]))
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            ctx.log(traceback.format_exc())

        if error:
            status = "failed"
        elif ctx.stop_reason == "user":
            status = "cancelled"
        elif ctx.stop_reason == "shutdown":
            status, error = "interrupted", "The app was closed while this job was running."
        else:
            status = "succeeded"
        with conn:
            conn.execute(
                f"""UPDATE job SET status = ?, result = ?, error = ?, done = ?, total = ?, message = ?,
                       finished_at = {db.NOW}
                    WHERE id = ? AND status = 'running'""",
                (status, json.dumps(result) if result is not None else None, error,
                 ctx.done, ctx.total_count, ctx.message, job["id"]),
            )
