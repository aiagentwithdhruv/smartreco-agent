"""Application settings, loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the app has. Defaults are safe for local development."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mesh API — the only LLM gateway used anywhere in this project.
    mesh_api_key: str = ""
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    mesh_chat_model: str = "openai/gpt-4o-mini"
    mesh_embedding_model: str = "openai/text-embedding-3-small"

    # App
    session_secret: str = "dev-secret-change-me"
    database_url: str = "sqlite:///./smartreco.sqlite3"
    chroma_dir: str = "./.chroma"

    # Trigger engine
    trigger_min_events: int = 8
    trigger_cooldown_seconds: int = 300
    trigger_staleness_minutes: int = 30
    retrieval_top_k: int = 6

    @property
    def mesh_configured(self) -> bool:
        """True when a real Mesh key is present, so live calls are possible."""
        return bool(self.mesh_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (tests clear the cache when they need to)."""
    return Settings()
