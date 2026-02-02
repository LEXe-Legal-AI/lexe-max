"""LEXe API Configuration.

All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_prefix="LEXE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8020
    log_level: str = "INFO"
    debug: bool = False

    # Database
    database_url: str = "postgresql://lexe:lexe@localhost:5433/lexe"
    kb_database_url: str | None = None  # Separate DB for KB massime (uses database_url if not set)

    # Redis/Valkey Cache
    redis_url: str = "redis://localhost:6380/0"

    # LiteLLM (for embeddings)
    litellm_api_base: str = "http://lexe-litellm:4000"
    litellm_api_key: str = ""

    # Cache Settings
    cache_ttl_hours: int = 24
    document_expire_days: int = 30

    # Health Monitoring
    health_failure_threshold: int = 5
    health_alert_cooldown_hours: int = 1
    admin_emails: str = ""
    alert_webhook_url: str = ""

    # HTTP Client
    http_timeout_seconds: int = 30
    http_max_retries: int = 3
    http_retry_delay_seconds: int = 1

    # Rate Limiting (requests per minute)
    rate_limit_normattiva: int = 30
    rate_limit_eurlex: int = 60
    rate_limit_brocardi: int = 30

    # Feature Flags
    ff_normattiva_enabled: bool = True
    ff_eurlex_enabled: bool = True
    ff_infolex_enabled: bool = True

    @property
    def admin_email_list(self) -> list[str]:
        """Parse admin emails into a list."""
        if not self.admin_emails:
            return []
        return [e.strip() for e in self.admin_emails.split(",") if e.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()
