# AGENTS.md: guide for AI agents working on babasu

babasu is a local computer vision platform: a desktop app where people bring their images and
go all the way to a deployed model. They annotate with AI help, run workflows, train, look at
results and export, all on their own machine. An agent answers questions about the data, sets
things up and makes bulk changes through the same API as the UI. Every decision here is judged
by what is best for the people using the app.

The app is FastAPI + Jinja, served by uvicorn on localhost and shown in a pywebview window, with
SQLite for app data. It is packaged with PyInstaller. Each step uses a library we maintain:

| Library   | What babasu uses it for                                         |
|-----------|-----------------------------------------------------------------|
| mayaku    | training, evaluation, export                                    |
| mozo      | models (auto-annotation, segmentation, tracking), workflows     |
| pixelflow | one result format, drawing predictions and annotations          |
| markinjs  | the annotation editor                                           |

## Ownership

One agent session maintains each project, and only that session changes its repo.
You are the owner and long-term maintainer of babasu, and its design is up to you.

- You may read the other repos (mayaku, mozo, pixelflow, markinjs), for example to see what
  they offer. Never edit them, commit in them, or run git commands that write there.
- Find another project's session with `ListAgents`.
- babasu depends only on released packages from PyPI, never on a local checkout's unreleased
  code.

## Fix it where it belongs

When babasu needs something a library doesn't have, it goes into that library, not around it
in babasu. Ask the owning session with `SendMessage`. Never copy a library's code into babasu or
patch around a missing feature.

A request says:

1. **What** is needed, as babasu would call it (`model.export("coreml", int8=True)`, not
   "Core ML support").
2. **Why**: the problem it solves for that library's users, not only for babasu.
3. **Proposed API**, if any, and what babasu does today without it.

Models come only through mozo. A question about a model (formats, memory, speed) is answered
from mozo (its manifest, its code, measurements with its classes), not from papers or Hugging
Face uploads. A missing model or format is a request to mozo.

## Principles

- **Proper, simple solutions.** No hacks, no workarounds, no over-engineering. Build what the
  current feature needs.
- **Server-rendered.** Jinja renders pages on the server. No client-side app, no client data
  loading, no fancy animations.
- **Minimal JS.** HTML and our own CSS first, small hand-written JS where a page needs it. A JS
  library is fine when it does exactly the one job needed (a markdown renderer, image zoom);
  vendor its build in `app/static/vendor/` with the version in the file name. No frameworks.
- **API first.** Anything the UI shows, the API can return. That is what lets the agent answer
  questions and set things up.
- **Local.** Data, models and the app stay on the user's machine.

## The UI

- Pages are built from the Jinja macros in `app/templates/components.html`, imported as `ui`
  (`{% import "components.html" as ui %}`).
- A component shows everything it is given. To hide something, leave it out: no show/hide
  flags, no cutting text with "…".
- API answers are clean: no keys or nesting without a reason, one name per thing, raw values
  (ISO times, bytes). Formatting for people happens in the components.
- The macOS window keeps its native title bar. A custom one broke dragging across displays and
  full screen.

## The agent

- It works through the app's API and its tools; it does not write pages or markup.
- It has to work well on small local models (gemma4:e2b class). Design for that by judgment:
  clear tool results, errors that say how to correct, plain data.
- Never tune a prompt, tool description or doc for one model. A change is made only when every
  model benefits from it (better data in a result, a missing tool or endpoint, clearer docs).
- The agent's own docs are `app/docs/*.md` (read through its `read_docs` tool). They are not
  the website docs.
- Model benchmarks run local models. Don't include Claude models unless the user asks: the test
  and its expected answers were written by Claude, so it would grade itself.

## Workspace data

Until the first release, everything in the workspace (projects, images, annotations,
conversations, runs) is test data. When a format changes, change it cleanly: no fallbacks or
compatibility code for existing workspace data. The exception is the user's `settings.json`: it
holds real API keys. Never print it.

Tests never touch the real workspace: `tests/conftest.py` points `HOME` at a temporary folder
before the app is imported, and gives each test an empty workspace.

## Rules

- Never commit, push, tag or release without the user asking. Commit as the user only: no
  `Co-Authored-By` lines and no mention of Claude. Small commits, one related change each.
- Before committing feature work, run `/simplify` on its changes and verify again.
- New dependencies need the user's approval. Bump pinned tools deliberately, never as a side
  effect.
- Numbers in docs, the README and commit messages are measured, with the model, data and
  hardware they came from.

## Docs

The website docs will live in `docs/` and be published on https://emrah.fr/babasu/. Before
writing or changing anything in `docs/`, read `DOCS-GUIDE.md` and follow its rules.

## Run

```bash
uv sync                                     # Python 3.12 venv with the dependencies
uv run python run.py                        # the desktop window
uv run uvicorn app.server:app --reload      # in a browser, reloading on changes
```

## Before you finish

```bash
uv run pytest -q
```

For a UI change, also open the page in the app and look at it. Show the test result; don't just
say it passed.

## Stop and ask the user

- New dependencies, or a JS library.
- Changes to the data model (new tables) or to how the agent's tools are shaped.
- Bringing back anything the user removed (a custom title bar, agent-written pages).
- Commits, pushes, tags, releases, version numbers.
- Deleting or rewriting files or commits you did not make.
