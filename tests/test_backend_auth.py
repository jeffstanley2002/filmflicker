from uuid import uuid4

import jwt
import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend import auth
from backend.config import get_settings


@pytest.fixture(autouse=True)
def auth_environment(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    auth._jwk_client_cache.clear()
    yield
    get_settings.cache_clear()


def credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def token_for(subject: str, issuer: str = "https://project.supabase.co/auth/v1") -> str:
    return jwt.encode(
        {"sub": subject, "aud": "authenticated", "iss": issuer},
        "test-secret-at-least-32-characters-long",
        algorithm="HS256",
    )


def test_valid_supabase_token_returns_uuid():
    subject = str(uuid4())
    assert auth.get_current_user_id(credentials(token_for(subject))) == subject


def test_attacker_controlled_issuer_is_rejected():
    with pytest.raises(HTTPException) as exc:
        auth.get_current_user_id(credentials(token_for(str(uuid4()), "https://attacker.example/auth/v1")))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired token"


def test_non_uuid_subject_is_rejected():
    with pytest.raises(HTTPException) as exc:
        auth.get_current_user_id(credentials(token_for("not-a-user-id")))
    assert exc.value.status_code == 401


def test_asymmetric_jwks_url_comes_only_from_configuration(monkeypatch):
    requested_urls = []

    class FakeClient:
        def __init__(self, url):
            requested_urls.append(url)

        def get_signing_key_from_jwt(self, token):
            raise jwt.InvalidTokenError("stop after URL assertion")

    monkeypatch.setattr(auth, "PyJWKClient", FakeClient)
    import json
    from jwt.utils import base64url_encode

    forged = ".".join(
        part.decode()
        for part in (
            base64url_encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()),
            base64url_encode(json.dumps({"sub": str(uuid4()), "iss": "https://attacker.example/auth/v1"}).encode()),
            base64url_encode(b"forged-signature"),
        )
    )
    with pytest.raises(HTTPException):
        auth.get_current_user_id(credentials(forged))
    assert requested_urls == ["https://project.supabase.co/auth/v1/.well-known/jwks.json"]
