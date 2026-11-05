"""Service configuration, read from environment variables (or a .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    model_path: str = "models/churn_model.joblib"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_s: float = 60.0
    ollama_keep_alive: str = "10m"
    llm_max_tokens: int = 256

    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
