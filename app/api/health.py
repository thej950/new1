from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Simple readiness endpoint for the platform foundation."""

    return {"status": "ok"}
