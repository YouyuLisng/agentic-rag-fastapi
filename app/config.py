from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    voyage_api_key: str
    database_url: str

    # Model routing: every per-turn tool-selection decision uses the fast
    # model; the final answer is escalated to the smart model only when
    # tools were actually used (a trivial no-tool answer stays on the
    # fast model -- no need to pay for Sonnet to say "hello back").
    claude_model_fast: str = "claude-haiku-4-5"
    claude_model_smart: str = "claude-sonnet-5"
    voyage_embedding_model: str = "voyage-3"
    voyage_embedding_dimensions: int = 1024

    max_agent_turns: int = 6


@lru_cache
def get_settings() -> Settings:
    # Fields are populated from the environment / .env at runtime by
    # pydantic-settings, not passed as constructor args -- pyright can't
    # see that statically.
    return Settings()  # pyright: ignore[reportCallIssue]
