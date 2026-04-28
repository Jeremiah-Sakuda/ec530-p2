"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Reads from .env file if present. All settings can be overridden
    via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Redis configuration
    redis_url: str = "redis://localhost:6379"

    # MongoDB configuration
    mongodb_uri: str = "mongodb://localhost:27017"
    db_name: str = "ec530_p2"

    # Use TinyDB instead of MongoDB (for offline dev)
    use_tinydb: bool = False
    tinydb_path: str = "data/tinydb"

    # Service URLs
    upload_service_url: str = "http://localhost:8001"
    inference_service_url: str = "http://localhost:8002"
    annotation_service_url: str = "http://localhost:8003"
    embedding_service_url: str = "http://localhost:8004"
    vector_index_service_url: str = "http://localhost:8005"
    query_service_url: str = "http://localhost:8006"

    # Vector settings
    vector_dim: int = 128

    # FAISS index path
    faiss_index_path: str = "data/faiss.index"
    faiss_id_map_path: str = "data/faiss_id_map.json"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload settings.
    """
    return Settings()
