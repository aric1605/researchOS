"""Configuration management using Pydantic Settings."""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import os


class Settings(BaseSettings):
    # LLM Settings
    LLM_PROVIDER: Literal["gemini", "groq"] = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Embedding Settings
    EMBEDDING_MODEL: str = "text-embedding-004"

    # Vector DB (Qdrant)
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "researchos_papers"

    # Relational DB (Neon Postgres)
    DATABASE_URL: Optional[str] = None

    # Access control & limits
    ACCESS_CODE: Optional[str] = None
    MAX_LLM_CALLS_PER_RUN: int = 25
    MAX_PAPERS_DEFAULT: int = 6
    RATE_LIMIT_RUNS_PER_HOUR: int = 5
    MAX_QUEUE_SIZE: int = 3

    # Server settings
    PORT: int = 8000
    BASE_URL: str = "http://localhost:8000"

    # Cache directory
    CACHE_DIR: str = ".cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("BASE_URL")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("DATABASE_URL")
    @classmethod
    def clean_database_url(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        # Neon pooled strings: remove channel_binding if present to avoid driver issues
        if "channel_binding=" in v:
            import re
            v = re.sub(r"[?&]channel_binding=[^&]*", "", v)
            if "?" not in v and "&" in v:
                v = v.replace("&", "?", 1)
        return v


settings = Settings()
