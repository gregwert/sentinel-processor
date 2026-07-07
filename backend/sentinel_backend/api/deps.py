"""FastAPI dependencies: session validation and lookup."""
import re
from fastapi import HTTPException
from sentinel_backend.storage import SessionStorage

_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def validate_uuid(value: str, label: str = "id") -> str:
    """Validate that value matches the standard UUID4 hex format.

    Args:
        value: The string to validate.
        label: Human-readable field name included in the 400 error detail;
            default ``"id"``.

    Returns:
        The original value, unchanged, when it matches the UUID pattern.

    Raises:
        HTTPException: 400 when value does not match the UUID4 hex pattern.
    """
    if not _UUID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label} format")
    return value


def get_session(session_id: str) -> SessionStorage:
    """Validate session_id format and return the live SessionStorage.

    Args:
        session_id: UUID string identifying the session to look up.

    Returns:
        The SessionStorage instance for the requested session.

    Raises:
        HTTPException: 400 when session_id is not a valid UUID.
        HTTPException: 404 when no session with that id exists.
    """
    validate_uuid(session_id, "session_id")
    try:
        return SessionStorage.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


def get_session_with_grid(session_id: str) -> SessionStorage:
    """Like get_session but also asserts a chip grid exists.

    Args:
        session_id: UUID string identifying the session to look up.

    Returns:
        The SessionStorage instance, guaranteed to have a saved chip grid.

    Raises:
        HTTPException: 400 when session_id is not a valid UUID.
        HTTPException: 404 when no session with that id exists.
        HTTPException: 422 when the session exists but has no chip grid yet.
    """
    s = get_session(session_id)
    if not s.grid_exists():
        raise HTTPException(status_code=422, detail="Chip grid not computed. Run PUT /chip-grid first.")
    return s
