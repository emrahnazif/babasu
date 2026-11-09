# babasu

Desktop app: FastAPI + Jinja templates served by uvicorn on localhost, shown in a native pywebview window.

## Setup

```sh
uv sync            # creates .venv with Python 3.12 and installs deps
```

## Run

```sh
uv run python run.py                          # native window (1280×800, min 900×600)
uv run uvicorn app.server:app --reload        # browser dev with auto-reload
```

## Layout

```
run.py                  entry point (dev + PyInstaller)
app/
  config.py             window sizes, host, resource paths (PyInstaller-aware)
  server.py             FastAPI app + background uvicorn server
  desktop.py            pywebview window
  templating.py         shared Jinja2Templates instance
  routes/pages.py       HTML routes
  templates/
    base.html           document shell
    pages/              one template per page
  static/css/
    main.css            entry; declares cascade layers and imports the rest
    reset.css tokens.css base.css layout.css
    components/         one stylesheet per component
```
