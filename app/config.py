from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    voyage_api_key: str
    database_url: str

    claude_model: str = "claude-sonnet-4-5"
    voyage_embedding_model: str = "voyage-3"
    voyage_embedding_dimensions: int = 1024

    max_agent_turns: int = 6


@lru_cache
def get_settings() -> Settings:
    # Fields are populated from the environment / .env at runtime by
    # pydantic-settings, not passed as constructor args -- pyright can't
    # see that statically.
    return Settings()  # pyright: ignore[reportCallIssue]
