"""Shared Jinja2 environment (single instance, imported by every page router)."""

from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def rupees(value: float) -> str:
    """Format a price the way the catalog displays it."""
    return f"₹{value:,.0f}"


templates.env.filters["rupees"] = rupees
