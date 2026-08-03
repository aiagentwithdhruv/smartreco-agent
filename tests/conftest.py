"""Test fixtures.

The environment is pointed at a throwaway directory *before* app modules are
imported, because app.db builds its engine at import time. No test ever touches
the developer's smartreco.sqlite3 or .chroma directory, and no test makes a
network call — the Mesh layer is always faked.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="smartreco-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.sqlite3"
os.environ["CHROMA_DIR"] = f"{_TMP}/chroma"
os.environ["SESSION_SECRET"] = "test-secret"
os.environ["MESH_API_KEY"] = ""  # never let a test reach the network

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import Base, SessionLocal, engine, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Product, User  # noqa: E402
from app.security import hash_password  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    """Every test starts from an empty schema."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db: Session) -> TestClient:
    """TestClient sharing the test's session, so assertions see request writes.

    Instantiated without a `with` block on purpose: that skips the lifespan and
    keeps init_db() from creating a database outside the temp directory.
    """
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def user(db: Session) -> User:
    u = User(email="user@example.com", pw_hash=hash_password("password123"), role="user")
    db.add(u)
    db.commit()
    return u


@pytest.fixture
def admin(db: Session) -> User:
    a = User(email="admin@example.com", pw_hash=hash_password("password123"), role="admin")
    db.add(a)
    db.commit()
    return a


@pytest.fixture
def products(db: Session) -> list[Product]:
    rows = [
        Product(
            title="Agentic Workflows with LangGraph",
            description="Stateful agents as explicit graphs.",
            category="Agentic AI",
            price=4999,
            level="intermediate",
            tags=["langgraph", "agents"],
        ),
        Product(
            title="SQL for Data Analysis",
            description="Joins, window functions and CTEs.",
            category="Data Analytics",
            price=2999,
            level="beginner",
            tags=["sql", "analytics"],
        ),
        Product(
            title="MLOps: Deploying and Monitoring Models",
            description="Ship a model and watch it drift.",
            category="MLOps",
            price=6999,
            level="intermediate",
            tags=["mlops", "docker"],
        ),
    ]
    db.add_all(rows)
    db.commit()
    return rows


def login(client: TestClient, email: str, password: str = "password123") -> None:
    """Log the client in; asserts the redirect so a broken login fails loudly."""
    response = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert response.status_code == 303, response.text
