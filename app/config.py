"""All runtime settings, read from environment variables or a local ``.env``.

Every cost guardrail is configurable here. ``ANTHROPIC_API_KEY`` is held as a
``SecretStr`` so it never appears in logs or reprs; without it the API still
serves retrieval, and ``/ask`` reports that answering is disabled.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_max_tokens: int = 4096
    anthropic_effort: str = "low"

    index_dir: Path = Path("data/index")
    answer_top_k: int = Field(6, ge=1, le=20, description="chunks sent to Claude per question")
    bm25_stemming: bool = False

    # Cost guardrails for /ask (the only endpoint that calls Claude).
    ask_rate_limit_per_hour: int = 10
    daily_question_cap: int = 100
    max_question_chars: int = 500

    # Abuse guardrails for the free endpoints (CPU only, no API cost).
    search_rate_limit_per_minute: int = 60
    upload_rate_limit_per_hour: int = 5

    # Temporary uploads: in memory only, never written to disk.
    upload_max_files: int = 5
    upload_max_file_bytes: int = 2 * 1024 * 1024
    upload_max_chunks: int = 200
    upload_ttl_seconds: int = 3600
    max_collections: int = 20

    # Number of reverse proxies in front of the API that append to
    # X-Forwarded-For. 0 = use the socket peer address (direct connections).
    forwarded_hops: int = 0
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @property
    def answering_enabled(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value())
