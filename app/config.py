from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    voyage_api_key: str
    database_url: str

    # Comma-separated -- the frontend's real deployed origin gets added
    # here via an env var once it has one, rather than hardcoded, since
    # this backend doesn't know in advance what platform/URL the
    # frontend ends up on. Defaults cover local dev only.
    cors_origins: str = "http://localhost:3000,http://localhost:3001"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # Model routing: every per-turn tool-selection decision uses the fast
    # model; the final answer is escalated to the smart model only when
    # tools were actually used (a trivial no-tool answer stays on the
    # fast model -- no need to pay for Sonnet to say "hello back").
    claude_model_fast: str = "claude-haiku-4-5"
    claude_model_smart: str = "claude-sonnet-5"
    voyage_embedding_model: str = "voyage-3"
    voyage_embedding_dimensions: int = 1024

    # Two-stage retrieval: the vector search pulls a wide candidate set
    # (rerank_candidate_count), then a Voyage cross-encoder reranks it
    # down to the caller's match_count. A cross-encoder reads
    # query+document together, so it separates near-duplicate distractors
    # that pure embedding cosine ranks almost tied -- the exact failure
    # mode behind the eval set's one real miss (疫情取消 -> force-majeure
    # vs cancellation). On this 8-document knowledge base the first stage
    # already ceilings (accuracy@5 = 100%), so rerank barely moves the
    # aggregate -- its payoff is at scale, where the candidate set is
    # wide and first-stage ranking mistakes are common. Toggleable so the
    # eval can measure with/without, and so it can be turned off to save
    # the extra API call where the headroom isn't worth it.
    rerank_enabled: bool = True
    voyage_rerank_model: str = "rerank-2.5-lite"
    rerank_candidate_count: int = 20

    # search_tours caps how many matching rows it returns, and reports the
    # true total alongside them. With 8 seed tours this never bites, but a
    # broad filter against a real catalog ("Asia tours") can match
    # hundreds -- returning them all floods the agent's context (cost,
    # truncation, the model picking arbitrarily from a wall of rows). The
    # cap keeps the result bounded and lets the agent tell the user to
    # narrow their filters when total > returned, instead of silently
    # dumping everything.
    tour_search_limit: int = 10

    max_agent_turns: int = 6

    # Rate limiting: every /chat and /documents request costs real Anthropic/
    # Voyage API money, and both are reachable by anyone with the frontend
    # URL -- these caps exist to bound worst-case spend from a single client,
    # not to enforce a business quota.
    rate_limit_chat: str = "10/minute"
    rate_limit_documents: str = "5/minute"
    rate_limit_eval: str = "3/minute"
    # Generation eval runs a full agent turn plus two judge calls per
    # case (~24 cases) -- far more expensive per hit than retrieval eval,
    # so it gets a tighter cap.
    rate_limit_eval_generation: str = "2/minute"


@lru_cache
def get_settings() -> Settings:
    # Fields are populated from the environment / .env at runtime by
    # pydantic-settings, not passed as constructor args -- pyright can't
    # see that statically.
    return Settings()  # pyright: ignore[reportCallIssue]
