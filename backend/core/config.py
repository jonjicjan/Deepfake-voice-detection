"""VoiceShield AI — Environment configuration."""

import os
from functools import lru_cache


class Settings:
    def __init__(self):
        self.debug: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
        self.api_key: str = os.getenv("VOICESHIELD_API_KEY", "").strip()
        self.admin_key: str = (
            os.getenv("VOICESHIELD_ADMIN_KEY", "").strip() or self.api_key
        )
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if o.strip()
        ]
        self.max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
        self.rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
        self.enable_docs: bool = os.getenv("ENABLE_DOCS", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
