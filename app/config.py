"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the app has. Defaults are safe for local development."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mesh API — the only LLM gateway used anywhere in this project.
    mesh_api_key: str = ""
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    # minimax/m2-her is one of the three models Mesh serves free of charge, so the
    # app has a working LLM on a zero-balance account. MESH_MODEL is an alias.
    # The field name is listed alongside the env aliases on purpose: a
    # validation_alias otherwise *replaces* the field name, and
    # Settings(mesh_chat_model=...) would be silently ignored.
    mesh_chat_model: str = Field(
        default="minimax/m2-her",
        validation_alias=AliasChoices("MESH_MODEL", "MESH_CHAT_MODEL", "mesh_chat_model"),
    )
    # Mesh has no free embedding model (checked 4 Aug 2026: 997 models, 3 free,
    # none of them embeddings), so this only works on a topped-up account.
    mesh_embedding_model: str = "google/embeddinggemma-300m"
    # Embeddings run locally by default — see app/services/embeddings.py for why
    # that is both the rule-compliant and the working choice. `mesh` and `auto`
    # are there for a funded key.
    embeddings: Literal["auto", "mesh", "local", "hashing"] = "local"

    # App
    session_secret: str = "dev-secret-change-me"
    database_url: str = "sqlite:///./smartreco.sqlite3"
    chroma_dir: str = "./.chroma"

    # Trigger engine
    trigger_min_events: int = 8
    trigger_cooldown_seconds: int = 300
    trigger_staleness_minutes: int = 30
    retrieval_top_k: int = 6

    # Proactive digest. Disabled by default so local/test runs never send mail.
    digest_enabled: bool = False
    digest_hour: int = Field(default=18, ge=0, le=23)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True

    # LangSmith is optional. LangGraph runs locally whether this is set or not.
    langchain_api_key: str = ""

    @property
    def mesh_configured(self) -> bool:
        """True when a real Mesh key is present, so live calls are possible."""
        return bool(self.mesh_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (tests clear the cache when they need to)."""
    return Settings()
