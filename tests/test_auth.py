"""Auth: registration, login, session cookie integrity, role gates."""

from sqlalchemy import select

from app.models import User
from app.security import SESSION_COOKIE, hash_password, read_session, sign_session, verify_password
from tests.conftest import login


def test_register_creates_user_and_signs_them_in(client, db):
    response = client.post(
        "/register",
        data={"email": "New@Example.com ", "password": "password123"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"

    user = db.scalar(select(User).where(User.email == "new@example.com"))
    assert user is not None, "email should be normalised to lowercase and trimmed"
    assert user.role == "user"
    assert user.pw_hash != "password123", "password must never be stored in plaintext"
    assert read_session(response.cookies[SESSION_COOKIE]) == user.id


def test_register_rejects_short_password(client, db):
    response = client.post("/register", data={"email": "a@b.com", "password": "short"})
    assert response.status_code == 400
    assert db.scalar(select(User).where(User.email == "a@b.com")) is None


def test_register_rejects_duplicate_email(client, user):
    response = client.post("/register", data={"email": user.email, "password": "password123"})
    assert response.status_code == 400
    assert "already registered" in response.text


def test_login_with_wrong_password_is_rejected(client, user):
    response = client.post("/login", data={"email": user.email, "password": "nope"}, follow_redirects=False)
    assert response.status_code == 401
    assert SESSION_COOKIE not in response.cookies


def test_login_then_logout_clears_the_session(client, user):
    login(client, user.email)
    assert user.email in client.get("/").text

    client.post("/logout", follow_redirects=False)
    assert user.email not in client.get("/").text


def test_tampered_cookie_is_not_trusted(client, user):
    login(client, user.email)
    client.cookies.set(SESSION_COOKIE, sign_session(user.id) + "x")
    assert user.email not in client.get("/").text


def test_cookie_signed_with_another_secret_is_rejected():
    from itsdangerous import URLSafeSerializer

    forged = URLSafeSerializer("not-our-secret", salt="smartreco-session").dumps({"uid": 1})
    assert read_session(forged) is None


def test_password_hashing_roundtrip():
    h = hash_password("password123")
    assert verify_password("password123", h)
    assert not verify_password("password124", h)
    assert not verify_password("password123", "not-a-hash")


def test_admin_pages_are_closed_to_regular_users(client, user, admin):
    """The admin gate is enforced by role, not by knowing the URL."""
    from app.deps import require_admin_page
    from fastapi import HTTPException

    try:
        require_admin_page(user)
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("a non-admin must not pass require_admin_page")

    assert require_admin_page(admin) is admin
