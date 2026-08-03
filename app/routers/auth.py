"""Registration, login, logout — form posts, signed session cookie."""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.deps import CurrentUser, DbSession
from app.models import User
from app.security import SESSION_COOKIE, hash_password, sign_session, verify_password
from app.templating import templates

router = APIRouter(tags=["auth"])


def _set_session(response: Response, user_id: int) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        sign_session(user_id),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 14,
    )


@router.get("/login")
def login_form(request: Request, user: CurrentUser):
    if user is not None:
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request, "login.html", {"user": None, "error": None})


@router.post("/login")
def login(
    request: Request,
    db: DbSession,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None or not verify_password(password, user.pw_hash):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"user": None, "error": "Wrong email or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    _set_session(response, user.id)
    return response


@router.get("/register")
def register_form(request: Request, user: CurrentUser):
    if user is not None:
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request, "register.html", {"user": None, "error": None})


@router.post("/register")
def register(
    request: Request,
    db: DbSession,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    email = email.strip().lower()
    if len(password) < 8:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"user": None, "error": "Password must be at least 8 characters."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if db.scalar(select(User).where(User.email == email)) is not None:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"user": None, "error": "That email is already registered."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    user = User(email=email, pw_hash=hash_password(password), role="user")
    db.add(user)
    db.commit()

    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    _set_session(response, user.id)
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(SESSION_COOKIE)
    return response
