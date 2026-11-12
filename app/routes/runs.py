"""Workflows, as JSON."""

from fastapi import APIRouter, HTTPException

from app import runs

router = APIRouter(prefix="/api")


@router.get("/workflows")
def api_workflows() -> list[dict]:
    """Every workflow: id, name, whether it's built in, and its models (or why it doesn't load)."""
    return runs.workflows()


@router.get("/workflows/{workflow_id}")
def api_workflow(workflow_id: str) -> dict:
    """The workflow's graph, the models it uses (licence, bytes left to download) and any missing files."""
    try:
        workflow = runs.load(workflow_id)
    except (KeyError, ValueError) as e:  # mozo's refusal of the file
        raise HTTPException(422, f"The workflow doesn't load: {e}") from None
    if workflow is None:
        raise HTTPException(404, "Workflow not found")
    return runs.detail(workflow_id, workflow)
