import json
import threading
import time

from app import jobs


@jobs.kind("test_until_cancelled", lane="io", title="Until cancelled")
def _until_cancelled(ctx: jobs.Context) -> None:
    while not ctx.cancelled:
        time.sleep(0.01)


@jobs.kind("test_fails", lane="gpu", title="Fails")
def _fails(ctx: jobs.Context) -> None:
    raise ValueError("no such model")


@jobs.kind("test_after", lane="io", title="After", needs=lambda conn, params: params["after"])
def _after(ctx: jobs.Context, after: list[int]) -> dict:
    return {"ran": True}


def test_cancel_stops_a_running_job(conn, workers, finished):
    job_id = jobs.start(conn, "test_until_cancelled", {})
    next(j for j in jobs.watched(conn, job_id, threading.Event(), seconds=10) if j["status"] != "queued")
    jobs.cancel(conn, job_id)
    assert finished(job_id)["status"] == "cancelled"


def test_a_job_waiting_for_one_that_fails_fails_too_and_says_why(conn, workers, finished):
    first = jobs.start(conn, "test_fails", {})
    waiting = jobs.start(conn, "test_after", {"after": [first]})
    job = finished(waiting)
    assert job["status"] == "failed"
    assert f"job {first}" in job["error"] and "no such model" in job["error"]


def test_jobs_left_queued_or_running_by_a_crash_are_interrupted_at_start(conn, finished):
    with conn:
        for status in ("queued", "running"):
            conn.execute("INSERT INTO job (kind, params, status) VALUES ('test_after', ?, ?)",
                         (json.dumps({"after": []}), status))
    jobs.run_workers()
    try:
        recent = jobs.list_recent(conn)
        assert {j["status"] for j in recent} == {"interrupted"}  # nothing restarts by itself
        again = jobs.retry(conn, recent[0]["id"])
        assert finished(again)["result"] == {"ran": True}
    finally:
        jobs.shutdown()
