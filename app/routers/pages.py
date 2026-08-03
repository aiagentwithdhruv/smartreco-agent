"""Browsable pages: catalog, search, product detail, cart.

Two signals are recorded server-side rather than by tracker.js because they are
deliberate actions, not passive browsing: `search` (a full page load carrying
the query) and `cart` (a POST). Everything passive — view, dwell, click — is
batched by static/tracker.js.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import or_, select

from app.deps import CurrentUser, DbSession, require_user_page
from app.models import Product, User
from app.services.tracking import record_event
from app.templating import templates

router = APIRouter(tags=["pages"])


def _categories(db: DbSession) -> list[str]:
    return sorted(db.scalars(select(Product.category).distinct()).all())


@router.get("/")
def home(
    request: Request,
    db: DbSession,
    user: CurrentUser,
    q: Annotated[str | None, Query(max_length=120)] = None,
    category: Annotated[str | None, Query(max_length=64)] = None,
):
    """Catalog + search. A non-empty `q` from a signed-in user is a high-intent signal."""
    stmt = select(Product).order_by(Product.id)
    q = (q or "").strip()
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Product.title.ilike(like), Product.description.ilike(like)))
    if category:
        stmt = stmt.where(Product.category == category)
    products = list(db.scalars(stmt).all())

    if q and user is not None:
        record_event(db, user_id=user.id, type="search", query=q)
        db.commit()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "products": products,
            "categories": _categories(db),
            "q": q,
            "active_category": category,
            "recommendation": None,  # filled in by the agent (S4)
        },
    )


@router.get("/products/{product_id}")
def product_detail(request: Request, db: DbSession, user: CurrentUser, product_id: int):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return templates.TemplateResponse(
        request,
        "product.html",
        {
            "user": user,
            "product": product,
            "categories": _categories(db),
            "recommendation": None,  # filled in by the agent (S4)
        },
    )


@router.post("/cart/add")
def add_to_cart(
    db: DbSession,
    user: Annotated[User, Depends(require_user_page)],
    product_id: Annotated[int, Form()],
):
    """Add-to-cart is the strongest intent signal short of checkout."""
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    record_event(db, user_id=user.id, type="cart", product_id=product.id, value=1.0)
    db.commit()
    return RedirectResponse(f"/products/{product_id}?added=1", status_code=status.HTTP_303_SEE_OTHER)
