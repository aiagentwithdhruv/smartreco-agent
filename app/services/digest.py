"""Daily delivery of already-stored recommendations.

The digest deliberately never invokes the recommendation agent: it reports the
latest stored result for users active in the last seven days and skips users who
do not have one. Log-only delivery is a legitimate local/default mode, not a
stub; it makes the scheduled behavior observable without credentials or cost.
"""

from __future__ import annotations

import logging
import smtplib
from collections.abc import Callable
from datetime import datetime, timedelta
from email.message import EmailMessage
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models import Event, Product, Recommendation, User, utcnow

log = logging.getLogger(__name__)
IST = "Asia/Kolkata"


def _latest_recommendation(db: Session, user_id: int) -> Recommendation | None:
    return db.scalar(
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.version.desc())
        .limit(1)
    )


def _message(db: Session, recommendation: Recommendation) -> str:
    product_ids = list(recommendation.product_ids or [])
    products = {
        product.id: product.title
        for product in db.scalars(select(Product).where(Product.id.in_(product_ids))).all()
    } if product_ids else {}
    titles = [products[product_id] for product_id in product_ids if product_id in products]
    course_line = f"\n\nRecommended courses: {', '.join(titles)}" if titles else ""
    return f"Your SmartReco recommendation:\n\n{recommendation.narrative}{course_line}"


def _send_telegram(settings: Settings, message: str) -> None:
    data = urlencode({"chat_id": settings.telegram_chat_id, "text": message}).encode()
    request = Request(
        f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
        data=data,
        method="POST",
    )
    with urlopen(request, timeout=20):  # noqa: S310 — fixed Telegram API host
        pass


def _send_email(settings: Settings, recipient: str, message: str) -> None:
    email = EmailMessage()
    email["Subject"] = "Your SmartReco recommendation"
    email["From"] = settings.smtp_from
    email["To"] = recipient
    email.set_content(message)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
        if settings.smtp_use_tls:
            smtp.starttls()
        if settings.smtp_username:
            smtp.login(settings.smtp_username, settings.smtp_password)
        smtp.send_message(email)


def deliver(user: User, message: str, settings: Settings) -> str:
    """Deliver one digest and return the selected channel name."""
    if settings.telegram_bot_token and settings.telegram_chat_id:
        _send_telegram(settings, message)
        return "telegram"
    if settings.smtp_host and settings.smtp_from:
        _send_email(settings, user.email, message)
        return "smtp"
    log.info("SmartReco digest for user_id=%s: %s", user.id, message)
    return "log"


def run_digest(
    *,
    session_factory: sessionmaker = SessionLocal,
    settings: Settings | None = None,
    now: datetime | None = None,
    delivery: Callable[[User, str, Settings], str] = deliver,
) -> list[int]:
    """Send stored recommendations to users with activity in the last 7 days.

    Returns the delivered user ids for observability and focused tests. Failures
    are isolated per user so one broken address cannot abort the whole job.
    """
    settings = settings or get_settings()
    cutoff = (now or utcnow()) - timedelta(days=7)
    delivered: list[int] = []

    with session_factory() as db:
        active_users = list(
            db.scalars(
                select(User)
                .where(exists().where(Event.user_id == User.id, Event.ts >= cutoff))
                .order_by(User.id)
            ).all()
        )
        for user in active_users:
            recommendation = _latest_recommendation(db, user.id)
            if recommendation is None:
                continue
            try:
                delivery(user, _message(db, recommendation), settings)
            except Exception:  # noqa: BLE001 — one delivery failure must not abort the batch
                log.exception("digest delivery failed for user_id=%s", user.id)
                continue
            delivered.append(user.id)
    return delivered


def build_digest_scheduler(
    *,
    settings: Settings | None = None,
    session_factory: sessionmaker = SessionLocal,
) -> BackgroundScheduler | None:
    """Build the daily IST scheduler, or nothing when the feature is disabled."""
    settings = settings or get_settings()
    if not settings.digest_enabled:
        return None

    scheduler = BackgroundScheduler(timezone=IST)
    scheduler.add_job(
        run_digest,
        CronTrigger(hour=settings.digest_hour, minute=0, timezone=IST),
        kwargs={"session_factory": session_factory, "settings": settings},
        id="daily-recommendation-digest",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    return scheduler
