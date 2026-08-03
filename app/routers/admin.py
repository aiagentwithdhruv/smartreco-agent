"""Admin catalog CRUD. Every write goes through app.services.catalog (dual-write)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.deps import DbSession, Store, require_admin_page
from app.models import Product, User
from app.services.catalog import ProductInput, create_product, delete_product, repair_sync, update_product
from app.templating import templates

router = APIRouter(prefix="/admin", tags=["admin"])

AdminUser = Annotated[User, Depends(require_admin_page)]
LEVELS = ("beginner", "intermediate", "advanced")


def _parse_form(title: str, description: str, category: str, price: float, level: str, tags: str) -> ProductInput:
    title = title.strip()
    category = category.strip()
    if not title or not category:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Title and category are required")
    if level not in LEVELS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Level must be one of {', '.join(LEVELS)}")
    if price < 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Price cannot be negative")
    return ProductInput(
        title=title,
        description=description.strip(),
        category=category,
        price=float(price),
        level=level,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )


@router.get("/products")
def list_products(request: Request, db: DbSession, store: Store, user: AdminUser):
    products = list(db.scalars(select(Product).order_by(Product.id)).all())
    return templates.TemplateResponse(
        request,
        "admin/products.html",
        {
            "user": user,
            "products": products,
            "indexed": store.count(),
            "unsynced": sum(1 for p in products if not p.vector_synced),
            "provider": store.provider.name,
            "indexed_provider": store.indexed_provider,
            "mismatch": store.provider_mismatch,
            "flash": request.query_params.get("flash"),
        },
    )


@router.get("/products/new")
def new_product_form(request: Request, user: AdminUser):
    return templates.TemplateResponse(
        request, "admin/product_form.html", {"user": user, "product": None, "levels": LEVELS}
    )


@router.post("/products")
def create(
    db: DbSession,
    store: Store,
    user: AdminUser,
    title: Annotated[str, Form()],
    category: Annotated[str, Form()],
    price: Annotated[float, Form()],
    level: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
):
    data = _parse_form(title, description, category, price, level, tags)
    product = create_product(db, data, store=store)
    return RedirectResponse(
        f"/admin/products?flash=Created+%23{product.id}", status_code=status.HTTP_303_SEE_OTHER
    )


@router.get("/products/{product_id}/edit")
def edit_product_form(request: Request, db: DbSession, user: AdminUser, product_id: int):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return templates.TemplateResponse(
        request, "admin/product_form.html", {"user": user, "product": product, "levels": LEVELS}
    )


@router.post("/products/{product_id}")
def update(
    db: DbSession,
    store: Store,
    user: AdminUser,
    product_id: int,
    title: Annotated[str, Form()],
    category: Annotated[str, Form()],
    price: Annotated[float, Form()],
    level: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    data = _parse_form(title, description, category, price, level, tags)
    update_product(db, product, data, store=store)
    return RedirectResponse(
        f"/admin/products?flash=Updated+%23{product_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@router.post("/products/{product_id}/delete")
def delete(db: DbSession, store: Store, user: AdminUser, product_id: int):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    delete_product(db, product, store=store)
    return RedirectResponse(
        f"/admin/products?flash=Deleted+%23{product_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@router.post("/sync/repair")
def repair(db: DbSession, store: Store, user: AdminUser):
    """Reconcile SQLite and the vector index, and say what changed."""
    report = repair_sync(db, store=store)
    flash = (
        f"Repair: {report.reindexed} reindexed, "
        f"{report.orphans_removed} orphans removed, {report.still_broken} still broken"
    )
    return RedirectResponse(
        f"/admin/products?flash={flash.replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER
    )
