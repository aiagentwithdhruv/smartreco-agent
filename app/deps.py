"""Shared FastAPI dependencies: current user, login gate, admin gate."""

from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.security import SESSION_COOKIE, read_session
from app.services.vector_store import VectorStore, get_vector_store

DbSession = Annotated[Session, Depends(get_db)]


def get_store() -> VectorStore:
    """The vector index, as a dependency so tests can inject their own."""
    return get_vector_store()


Store = Annotated[VectorStore, Depends(get_store)]


def get_current_user(
    db: DbSession,
    smartreco_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> User | None:
    """The signed-in user, or None. Never raises — pages render either way."""
    uid = read_session(smartreco_session)
    if uid is None:
        return None
    return db.get(User, uid)


CurrentUser = Annotated[User | None, Depends(get_current_user)]


def require_user(user: CurrentUser) -> User:
    """API gate: 401 when signed out."""
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in required")
    return user


def require_admin(user: Annotated[User, Depends(require_user)]) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user


def require_user_page(user: CurrentUser) -> User:
    """Page gate: bounce to the login form instead of showing a JSON 401."""
    if user is None:
        raise HTTPException(
            status.HTTP_303_SEE_OTHER,
            "Sign in required",
            headers={"Location": "/login"},
        )
    return user


def require_admin_page(user: Annotated[User, Depends(require_user_page)]) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user
