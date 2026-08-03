"""Password hashing and signed-cookie sessions.

Deliberately small: bcrypt via passlib for passwords, an itsdangerous-signed
cookie holding nothing but the user id. No JWTs, no server-side session store.
"""

from itsdangerous import BadSignature, URLSafeSerializer
from passlib.context import CryptContext

from app.config import get_settings

SESSION_COOKIE = "smartreco_session"

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, pw_hash: str) -> bool:
    try:
        return _pwd_context.verify(password, pw_hash)
    except ValueError:
        # Malformed/legacy hash in the row — treat as a failed login, not a 500.
        return False


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_settings().session_secret, salt="smartreco-session")


def sign_session(user_id: int) -> str:
    """Serialize a user id into the cookie value."""
    return _serializer().dumps({"uid": user_id})


def read_session(raw: str | None) -> int | None:
    """Return the user id in a cookie value, or None if absent/tampered."""
    if not raw:
        return None
    try:
        data = _serializer().loads(raw)
    except BadSignature:
        return None
    uid = data.get("uid") if isinstance(data, dict) else None
    return uid if isinstance(uid, int) else None
