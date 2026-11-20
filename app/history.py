"""History: what was done to the user's data in the last 30 days, so it can be undone.

SQLite triggers keep every write to the user's tables (db.HISTORY_TABLES) under the action that
made it: one request, job or agent turn (db.action). Files are kept too: a deleted image's file
stays in place while an entry can put its row back; a deleted workflow moves to the trash.
Not kept: settings.json (models, agents) and a saved workflow written over.
"""

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from app import config, db, media

DAYS = 30  # how long anything done can be undone
OLD = f"SELECT id FROM action WHERE at < strftime('{db.TIME}', 'now', '-{DAYS} days')"


class Refused(Exception):
    """An action that can't be undone now, and why."""


def trash(conn: sqlite3.Connection, path: Path) -> Path:
    """Move a file in the workspace to the trash, under the current action, which undoing
    moves it back; returns where it went. Caller commits (a move whose entry isn't committed is
    removed by purge)."""
    moved = config.TRASH_DIR / db.current_action().id / path.relative_to(config.DATA_DIR)
    moved.parent.mkdir(parents=True, exist_ok=True)
    path.rename(moved)
    _record_file(conn, "trash", path, moved)
    return moved


def _record_file(conn: sqlite3.Connection, op: str, path: Path, moved: Path) -> None:
    """A file's move in history, as the triggers record a row's write."""
    action = db.current_action()
    conn.execute("INSERT OR IGNORE INTO action (id, source, label, root) VALUES (?, ?, ?, ?)",
                 (action.id, action.source, action.label, action.id))
    conn.execute("INSERT INTO history (action_id, tbl, op, key, before) VALUES (?, 'file', ?, "
                 "json_object('path', ?), json_object('trash', ?))",
                 (action.id, op, *(p.relative_to(config.DATA_DIR).as_posix() for p in (path, moved))))


_columns_of: dict[str, set[str]] = {}  # the schema doesn't change once the app runs (db.migrate)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if table not in _columns_of:
        _columns_of[table] = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    return _columns_of[table]


def _where(key: dict) -> tuple[str, dict]:
    """A WHERE clause for a row's key, and its parameters."""
    return " AND ".join(f"{k} = :k_{k}" for k in key), {f"k_{k}": v for k, v in key.items()}


# --- Reading ---

# How the history page and the agent name what an action changed.
_NAMES = {"project": ("project", "projects"), "media": ("image", "images"), "file": ("file", "files")}
_DONE = {"insert": "added", "update": "changed", "delete": "deleted", "trash": "moved to the trash",
         "untrash": "back from the trash"}

# The last action of action `{0}`'s chain (action.root): the end of its line of undos and redos.
# An action is undone when that one is an odd number of undos deeper.
_LAST = "(SELECT id FROM action WHERE root = {0}.root ORDER BY depth DESC, rowid DESC LIMIT 1)"
# An action's entries (mine, of action m) joined to the later entries (h, of action ha) on the same
# rows and files, by other chains; not a chain begun after m and undone since: its changes and
# undos cancel out.
_LATER = """history mine JOIN action m ON m.id = mine.action_id
            JOIN history h ON h.tbl = mine.tbl AND h.key = mine.key AND h.id > mine.id
            JOIN action ha ON ha.id = h.action_id AND ha.root != m.root
            WHERE NOT ((SELECT max(depth) FROM action WHERE root = ha.root) % 2 = 1
                       AND coalesce((SELECT rowid FROM action WHERE id = ha.root), 0) > m.rowid)"""


def actions(conn: sqlite3.Connection, limit: int = 100, action_id: str | None = None) -> list[dict]:
    """The latest actions (or the one given), newest first: id, source (request, job, agent, app),
    kind (do, undo, redo), label, at, what it changed (in words), the undo that undid it or the
    action it undid, if any, and whether a later action changed the same things (`changed_since`:
    it can't be undone before that one is)."""
    which, params = ("WHERE id = ?", [action_id]) if action_id else ("", [])
    rows = conn.execute(
        f"""SELECT a.id, a.source, a.label, a.at, a.depth, last.id AS last, last.depth AS last_depth,
                   (SELECT id FROM action WHERE root = a.root AND depth = a.depth - 1 ORDER BY rowid DESC LIMIT 1) AS undoes,
                   EXISTS (SELECT 1 FROM {_LATER} AND mine.action_id = last.id) AS changed_since
            FROM (SELECT rowid AS n, * FROM action {which} ORDER BY rowid DESC LIMIT ?) a
            JOIN action last ON last.id = {_LAST.format("a")}
            ORDER BY a.n DESC""", (*params, limit)).fetchall()
    found = []
    for r in rows:
        kind = "do" if r["depth"] == 0 else "undo" if r["depth"] % 2 else "redo"
        undone = (r["last_depth"] - r["depth"]) % 2 == 1
        found.append({"id": r["id"], "source": r["source"], "kind": kind, "label": r["label"],
                      "at": r["at"], "changed": _changed(conn, r["id"]), "undoes": r["undoes"],
                      "undone_by": r["last"] if undone else None, "changed_since": bool(r["changed_since"])})
    return found


def _changed(conn: sqlite3.Connection, action_id: str) -> str:
    """What an action changed, per row: added (its first entry an insert), deleted (its last a
    delete) or changed; "project “Cats” deleted, 1,240 images deleted"."""
    groups = conn.execute(
        """WITH touched AS (SELECT tbl, key, min(id) AS first, max(id) AS last FROM history
                            WHERE action_id = ? GROUP BY tbl, key)
           SELECT t.tbl, CASE f.op WHEN 'insert' THEN 'insert' ELSE l.op END AS op, count(*) AS n,
                  max(f.key) AS key, max(f.before) AS before  -- the row's, when there's one
           FROM touched t JOIN history f ON f.id = t.first JOIN history l ON l.id = t.last
           GROUP BY 1, 2 ORDER BY min(t.first)""", (action_id,)).fetchall()
    words = []
    for g in groups:
        one, many = _NAMES.get(g["tbl"], (g["tbl"], g["tbl"]))
        name = _name(conn, g["tbl"], json.loads(g["key"]), json.loads(g["before"] or "{}")) if g["n"] == 1 else None
        what = f"{one} “{name}”" if name else one if g["n"] == 1 else f"{g['n']:,} {many}"
        words.append(f"{what} {_DONE[g['op']]}")
    return ", ".join(words)


def _name(conn: sqlite3.Connection, table: str, key: dict, before: dict) -> str | None:
    """What a row or file is called: its path, or its name as it is now (as it was, once deleted)."""
    if table == "file":
        return key["path"]
    if "name" not in _columns(conn, table):
        return None
    where, params = _where(key)
    row = conn.execute(f"SELECT name FROM {table} WHERE {where}", params).fetchone()
    return row["name"] if row else before.get("name")


# --- Undoing ---


def latest(conn: sqlite3.Connection, redo: bool = False) -> str | None:
    """The action to undo next, as an editor's Undo picks it: the newest done (or redone) and not
    undone, so undoing again goes further back. With `redo`, the one Redo undoes: the newest undo
    not undone, unless something new was done since. None if there's none."""
    rows = conn.execute("""SELECT id, depth, (SELECT max(depth) FROM action WHERE root = a.root) AS last_depth
                           FROM action a ORDER BY rowid DESC""")
    for r in rows:
        undoing = r["depth"] % 2 == 1
        if redo and r["depth"] == 0:
            return None  # something new was done: nothing to redo
        undone = (r["last_depth"] - r["depth"]) % 2 == 1
        if not undone and undoing == redo:
            return r["id"]
    return None


def undo(conn: sqlite3.Connection, action_id: str) -> dict:
    """Undo an action: in one transaction, its entries in reverse order (the inserted rows deleted,
    the old values written back, the deleted rows inserted, the trashed files moved back), as a new
    action that names it, which undoing redoes. Returns the new action as actions() shows it.
    LookupError if there's no such action; Refused if a later action changed the same rows (undo
    that first), or a row or file can't go back."""
    action = conn.execute(f"""SELECT a.label, a.depth, last.id AS last, last.root, last.depth AS last_depth
                              FROM action a JOIN action last ON last.id = {_LAST.format("a")} WHERE a.id = ?""",
                          (action_id,)).fetchone()
    if action is None:
        raise LookupError(f"No action {action_id} in the last {DAYS} days")
    if (action["last_depth"] - action["depth"]) % 2 == 1:
        raise Refused("It's undone already; undoing that undo does it again.")
    action_id = action["last"]  # a redo further on has the same effect: undo that, so the chain stays a line
    later = conn.execute(f"SELECT DISTINCT {_LAST.format('ha')} AS id FROM {_LATER} AND mine.action_id = ?",
                         (action_id,)).fetchall()  # each chain's last: undo that first
    if later:
        named = [actions(conn, action_id=r["id"])[0] for r in later]
        raise Refused("Changed again since by " + "; ".join(f"“{a['label']}” ({a['at']})" for a in named)
                      + ": undo that first.")

    moved: list[tuple[Path, Path]] = []  # files moved (where they are, where they were), put back if this fails
    with db.action(db.current_action().source, action["label"]):  # the kind (undo, redo) comes from its depth
        undoing = db.current_action()
        try:
            conn.execute("BEGIN")
            conn.execute("PRAGMA defer_foreign_keys = ON")  # rows go back in any order; checked at the commit
            conn.execute("INSERT INTO action (id, source, label, root, depth) VALUES (?, ?, ?, ?, ?)",
                         (undoing.id, undoing.source, undoing.label, action["root"], action["last_depth"] + 1))
            for entry in conn.execute("SELECT * FROM history WHERE action_id = ? ORDER BY id DESC", (action_id,)).fetchall():
                if entry["tbl"] == "file":
                    _undo_file(conn, entry, moved)
                else:
                    _undo_row(conn, entry)
            conn.commit()
        except Exception as e:
            conn.rollback()
            for now, was in reversed(moved):
                now.rename(was)
            if isinstance(e, sqlite3.IntegrityError):
                raise Refused(f"A row can't go back ({e}): something like it was made since.") from None
            raise
    return actions(conn, action_id=undoing.id)[0]


def _undo_row(conn: sqlite3.Connection, entry: sqlite3.Row) -> None:
    table = entry["tbl"]
    where, params = _where(json.loads(entry["key"]))
    if entry["op"] == "insert":
        conn.execute(f"DELETE FROM {table} WHERE {where}", params)
        return
    columns = _columns(conn, table)
    before = {k: v for k, v in json.loads(entry["before"]).items() if k in columns}  # a column dropped since is left out
    if entry["op"] == "update":
        sets = ", ".join(f"{k} = :{k}" for k in before)
        conn.execute(f"UPDATE {table} SET {sets} WHERE {where}", params | before)
    else:
        conn.execute(f"INSERT INTO {table} ({', '.join(before)}) VALUES ({', '.join(':' + k for k in before)})", before)


def _undo_file(conn: sqlite3.Connection, entry: sqlite3.Row, moved: list[tuple[Path, Path]]) -> None:
    path = config.DATA_DIR / json.loads(entry["key"])["path"]
    if entry["op"] == "untrash":  # it was brought back: back to the trash
        moved.append((trash(conn, path), path))
        return
    in_trash = config.DATA_DIR / json.loads(entry["before"])["trash"]
    if path.exists():
        raise Refused(f"{path.relative_to(config.DATA_DIR)} was made again since; delete it first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    in_trash.rename(path)
    moved.append((path, in_trash))
    _record_file(conn, "untrash", path, in_trash)


# --- Forgetting ---


def purge() -> None:
    """Forget actions older than DAYS with their entries (at start), then remove what only they
    could bring back: image files no media row or remaining entry names, and their trash."""
    with closing(db.connect()) as conn:
        with conn:
            images = conn.execute(f"""SELECT DISTINCT before ->> 'sha256', before ->> 'ext' FROM history
                                      WHERE tbl = 'media' AND op = 'delete' AND action_id IN ({OLD})""").fetchall()
            conn.execute(f"DELETE FROM action WHERE id IN ({OLD})")
        named = {r[0] for r in conn.execute("SELECT sha256 FROM media UNION SELECT before ->> 'sha256' FROM history "
                                            "WHERE tbl = 'media' AND before IS NOT NULL")}
        kept = {r[0] for r in conn.execute("SELECT id FROM action")}
    for sha256, ext in images:
        if sha256 not in named:
            media.forget(sha256, ext)
    if config.TRASH_DIR.is_dir():
        for folder in config.TRASH_DIR.iterdir():
            if folder.name not in kept:
                shutil.rmtree(folder)
