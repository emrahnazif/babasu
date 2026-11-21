"""Test the agents on requests whose answers are known, in the real workspace: afterwards the app
shows every conversation (titled with its result) and the projects the agents worked on.

    uv run python -m scripts.agent_evals --agents gemma4:e2b,gemma4:e4b      # 3 runs of each request
    uv run python -m scripts.agent_evals --agents gemma4:e2b --runs 1 --cases count-triangles,delete-circles

Each run of a request gets a project of its own, made for it ("Eval · gemma4:e2b · delete-circles
· 1": the Shapes images, with known annotations), and a conversation started on that project's
page, as the chat starts one. When the turn ends, a check reads the database and the reply (a
number in it), never another model's opinion. A table of the results is printed and written to
reports/.

The last run's projects stay to look at; the next run deletes them first (and the workflows they
saved), so they don't pile up in the app's lists. Their conversations stay.

Close the app first: its job workers and these would both run the same jobs. Model cases download
the default workflow's weights the first time.
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / "reports"


@dataclass
class Case:
    name: str
    request: str
    check: Callable  # (conn, project_id, reply, started_at) -> (passed, what was found)


# --- The project each run gets ---

def shapes() -> list[tuple[str, bytes, list[dict]]]:
    """The Shapes images (scripts/seed.py's, the same every time): file name, JPEG, annotations."""
    from scripts import seed

    return seed.shapes()


def make_project(conn, name: str) -> int:
    """A project of the Shapes images, with their annotations, which the checks count from."""
    from scripts import seed

    with conn:
        return seed.make_project(conn, name, "Made by scripts/agent_evals.py.")


def workflow_name(project_id: int) -> str:
    """The name the build-workflow request asks the workflow to be saved as."""
    return f"Squares {project_id}"


def objects(conn, project_id: int) -> list[dict]:
    """Every object in the project now, with its image's file name."""
    from app import projects

    ids = [r[0] for r in conn.execute("SELECT id FROM media WHERE project_id = ? ORDER BY id", (project_id,))]
    return [a | {"filename": image["filename"]} for image in projects.images(conn, ids)["images"]
            for a in image["annotations"]]


def numbers(text: str) -> set[int]:
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)}


# --- The checks ---

def says(count: Callable[[], int]) -> Callable:
    """The reply has the number `count` gives."""
    def check(conn, project_id, reply, started_at):
        expected = count()
        return expected in numbers(reply), f"expected {expected}, said {sorted(numbers(reply)) or 'no number'}"
    return check


def count_images(test: Callable) -> int:
    return sum(1 for _, _, annotations in shapes() if test(annotations))


def count_objects(test: Callable) -> int:
    return sum(1 for _, _, annotations in shapes() for a in annotations if test(a))


def classes_now(conn, project_id) -> dict[str, int]:
    from app import media

    return media.class_counts(objects(conn, project_id))


def classes_before() -> dict[str, int]:
    from app import media

    return media.class_counts([a for _, _, annotations in shapes() for a in annotations])


def became(expected: Callable) -> Callable:
    """The project's objects per class are what `expected(before)` says."""
    def check(conn, project_id, reply, started_at):
        want, now = expected(classes_before()), classes_now(conn, project_id)
        return now == want, f"expected {want}, now {now}"
    return check


def without_narrow(conn, project_id, reply, started_at):
    now = objects(conn, project_id)
    narrow = sum(1 for a in now if a["bbox"][2] < 80)
    want = count_objects(lambda a: a["bbox"][2] >= 80)
    return narrow == 0 and len(now) == want, f"expected {want} objects, none narrow; now {len(now)}, {narrow} narrow"


def most_objects(conn, project_id, reply, started_at):
    top = max(len(annotations) for _, _, annotations in shapes())
    names = [name for name, _, annotations in shapes() if len(annotations) == top]
    return any(n in reply or n.removesuffix(".jpg") in reply for n in names), f"expected one of {names}"


def ran_model(conn, project_id, reply, started_at):
    from app import runs

    ids = {r[0] for r in conn.execute("SELECT id FROM media WHERE project_id = ?", (project_id,))}
    jobs = [j for j in conn.execute("SELECT status, params FROM job WHERE kind = ? AND created_at >= ?",
                                    (runs.KIND, started_at)) if ids & set(json.loads(j[1]).get("media_ids", []))]
    found = sum(1 for a in objects(conn, project_id) if a.get("score") is not None)
    statuses = [j[0] for j in jobs]
    return "succeeded" in statuses, f"runs {statuses or 'none'}, {found} objects found by a model"


def saved_workflow(conn, project_id, reply, started_at):
    """A workflow saved under the name the request gave, during this run (so not an earlier run's)."""
    from app import config

    new = [json.loads(p.read_text()).get("name", p.stem) for p in config.WORKFLOWS_DIR.glob("*.json")
           if p.stat().st_mtime >= datetime.fromisoformat(started_at).timestamp()]
    wanted = workflow_name(project_id)
    return wanted.casefold() in {n.casefold() for n in new}, f"expected {wanted!r}, saved {new or 'none'}"


CASES = [
    Case("count-triangles", "How many images in this project have a triangle?",
         says(lambda: count_images(lambda an: any(o["class"] == "triangle" for o in an)))),
    Case("count-circles", "How many circles are there in this project, in all?",
         says(lambda: count_objects(lambda o: o["class"] == "circle"))),
    Case("count-no-square", "How many images here have no square?",
         says(lambda: count_images(lambda an: all(o["class"] != "square" for o in an)))),
    Case("most-objects", "Which image in this project has the most objects? Give me its file name.", most_objects),
    Case("delete-circles", "Delete all the circles in this project.",
         became(lambda before: {k: v for k, v in before.items() if k != "circle"})),
    Case("rename-class", "Rename the class triangle to pyramid in this project.",
         became(lambda before: {("pyramid" if k == "triangle" else k): v for k, v in before.items()})),
    Case("remove-narrow", "Remove every object narrower than 80 pixels in this project.", without_narrow),
    Case("find-squares", "Find the squares in 3 images of this project with a model.", ran_model),
    Case("build-workflow", "Build a workflow that finds squares and keeps them as annotations, try it on 2 images "
         "of this project, then save it as 'Squares {project}'.", saved_workflow),
]


# --- Running ---

def clear_last_run(conn) -> None:
    """Delete the projects an earlier run made, and the workflows saved for them ('Squares ID')."""
    from app import projects, runs

    made = [r[0] for r in conn.execute("SELECT id FROM project WHERE name LIKE 'Eval · %'")]
    for project_id in made:
        projects.remove(conn, project_id)
    saved = {workflow_name(project_id).casefold() for project_id in made}
    for workflow in runs.named("squares "):
        if workflow["name"].casefold() in saved:
            runs.delete_workflow(workflow["id"])
    print(f"Deleted the last run's {len(made)} projects.")


def app_running() -> bool:
    found = subprocess.run(["pgrep", "-f", r"run\.py|app\.server:app"], capture_output=True, text=True)
    return bool(found.stdout.strip())


def page_of(client, path: str) -> tuple[str, str]:
    """A page's title and its own words (the `about` meta tag), as the chat sends them."""
    from app.templating import page_name

    html = client.get(path).text
    title = page_name(unescape(re.search(r"<title>(.*?)</title>", html, re.DOTALL)[1]))
    about = re.search(r'<meta name="about" content="(.*?)">', html, re.DOTALL)
    return title, unescape(about[1]).strip() if about else ""


def run_case(conn, client, agent, model, case: Case, attempt: int) -> dict:
    from app import agent as agent_mode
    from app import db

    name = f"Eval · {model.name} · {case.name} · {attempt}"
    project_id = make_project(conn, name)
    page = f"/projects/{project_id}"
    title, about = page_of(client, page)
    conversation_id = agent_mode.new_conversation(conn)
    with conn:
        conn.execute("UPDATE conversation SET agent_id = ?, title = ? WHERE id = ?", (agent.id, name, conversation_id))
    started_at = conn.execute(f"SELECT {db.NOW}").fetchone()[0]
    started, reply, tools, tokens, status, error = time.monotonic(), [], 0, 0, "", ""
    request = case.request.format(project=project_id)  # a name of its own, where a request needs one
    for event in agent_mode.run_turn(conversation_id, request, threading.Event(), page, title, about):
        kind = event["type"]
        if kind == "text":
            reply.append(event["delta"])
        elif kind == "tool" and event["status"] == "running":
            tools += 1
        elif kind == "usage":
            tokens += (event.get("tokens_in") or 0) + (event.get("tokens_out") or 0)
        elif kind in ("done", "stopped", "error"):
            status, error = kind, event.get("message", "")
    seconds = time.monotonic() - started
    reply_text = "".join(reply)
    passed, found = case.check(conn, project_id, reply_text, started_at)
    if status != "done":
        passed, found = False, f"turn {status}{': ' + error if error else ''}; {found}"
    with conn:
        conn.execute("UPDATE conversation SET title = ? WHERE id = ?",
                     (f"{'✓' if passed else '✗'} {name}", conversation_id))
    return {"agent": model.name, "case": case.name, "attempt": attempt, "passed": passed, "found": found,
            "seconds": round(seconds, 1), "tools": tools, "tokens": tokens, "conversation": conversation_id,
            "project": project_id, "reply": reply_text.strip()}


def pick_agents(settings, wanted: list[str] | None):
    """The agents named (by name or model, as try_agent picks one), or all that have a model."""
    from scripts.try_agent import pick_agent

    if wanted:
        return [pick_agent(settings, name) for name in wanted]
    return [(a, m) for a in settings.get().agents if (m := settings.find_model(a.model_id))]


def report(results: list[dict]) -> str:
    lines = ["| agent | case | passed | seconds | tools | tokens | conversation | found |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['agent']} | {r['case']} #{r['attempt']} | {'✓' if r['passed'] else '✗'} | {r['seconds']} | "
              f"{r['tools']} | {r['tokens']:,} | {r['conversation']} | {r['found']} |" for r in results]
    lines += ["", "| agent | passed |", "|---|---|"]
    for agent in dict.fromkeys(r["agent"] for r in results):
        mine = [r for r in results if r["agent"] == agent]
        lines.append(f"| {agent} | {sum(r['passed'] for r in mine)} of {len(mine)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agents", help="Agents by name or model, comma-separated (default: all).")
    parser.add_argument("--cases", help=f"Comma-separated, of: {', '.join(c.name for c in CASES)} (default: all).")
    parser.add_argument("--runs", type=int, default=3, help="Runs of each request per agent.")
    args = parser.parse_args()
    if app_running():
        sys.exit("The app is running: close it first (its job workers would run these jobs too).")
    cases = [c for c in CASES if not args.cases or c.name in args.cases.split(",")]

    from fastapi.testclient import TestClient

    import app.server
    from app import db, jobs, settings, vision

    db.migrate()
    vision.setup()
    jobs.run_workers()
    client = TestClient(app.server.app)
    results = []
    try:
        with closing(db.connect()) as conn:
            clear_last_run(conn)
            for agent, model in pick_agents(settings, args.agents.split(",") if args.agents else None):
                for case in cases:
                    for attempt in range(1, args.runs + 1):
                        result = run_case(conn, client, agent, model, case, attempt)
                        results.append(result)
                        print(f"{'✓' if result['passed'] else '✗'} {model.name} {case.name} #{attempt}: "
                              f"{result['found']} ({result['seconds']} s, {result['tools']} tools)", flush=True)
    finally:
        jobs.shutdown()
        if results:
            REPORTS.mkdir(exist_ok=True)
            path = REPORTS / f"agent-evals-{datetime.now():%Y-%m-%d-%H%M}.md"
            path.write_text(report(results) + "\n")
            (path.with_suffix(".json")).write_text(json.dumps(results, indent=1, ensure_ascii=False))
            print("\n" + report(results).split("\n\n")[-1] + f"\n\nWritten to {path}")


if __name__ == "__main__":
    main()
