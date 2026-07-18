"""Validated runtime configuration shared by the API and auth layer."""
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


class Settings:
    def __init__(self) -> None:
        self.supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.supabase_jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "")
        self.database_url = os.environ.get("DATABASE_URL", "")
        self.environment = os.environ.get("APP_ENV", "development").lower()
        self.allowed_origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
            ).split(",")
            if origin.strip()
        )

    @property
    def issuer(self) -> str:
        return f"{self.supabase_url}/auth/v1"

    @property
    def jwks_url(self) -> str:
        return f"{self.issuer}/.well-known/jwks.json"

    def validate(self, *, require_database: bool = True) -> list[str]:
        errors = []
        parsed = urlparse(self.supabase_url)
        if not self.supabase_url or parsed.scheme != "https" or not parsed.netloc:
            errors.append("SUPABASE_URL must be a valid https URL")
        if not self.supabase_jwt_secret:
            errors.append("SUPABASE_JWT_SECRET is required")
        if require_database and not self.database_url:
            errors.append("DATABASE_URL is required")
        if not self.allowed_origins:
            errors.append("ALLOWED_ORIGINS must contain at least one origin")
        if "*" in self.allowed_origins:
            errors.append("ALLOWED_ORIGINS cannot use '*' with credentialed requests")
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
