"""Supabase JWT verification. The frontend authenticates directly against
Supabase Auth (signup/login/session refresh) and sends the resulting JWT
as `Authorization: Bearer <token>` on every request here. No password ever
touches this backend - it only verifies a token Supabase already issued.
"""
from uuid import UUID

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import get_settings

security = HTTPBearer()

_jwk_client_cache = {}


def _decode_with_jwks(token: str, alg: str) -> dict:
    """Verifies newer Supabase asymmetric JWTs using the project's JWKS.

    Supabase projects can issue access tokens signed by asymmetric keys
    instead of the legacy shared JWT secret. Those tokens advertise an
    algorithm such as ES256/RS256, so HS256-only verification rejects them
    with "The specified alg value is not allowed".
    """
    settings = get_settings()
    issuer = settings.issuer
    jwks_url = settings.jwks_url
    if jwks_url not in _jwk_client_cache:
        _jwk_client_cache[jwks_url] = PyJWKClient(jwks_url)
    signing_key = _jwk_client_cache[jwks_url].get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=[alg],
        audience="authenticated",
        issuer=issuer,
    )


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """FastAPI dependency: verifies the bearer token and returns the
    Supabase user's UUID (the JWT's `sub` claim). Raises 401 on anything
    invalid/expired/missing - never trusts an unverified claim."""
    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg")
        if alg == "HS256":
            settings = get_settings()
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=settings.issuer,
            )
        elif alg in {"RS256", "ES256"}:
            payload = _decode_with_jwks(token, alg)
        else:
            raise jwt.InvalidAlgorithmError(f"Unsupported JWT algorithm: {alg}")
    except jwt.exceptions.MissingCryptographyError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backend is missing cryptography. Run `pip install -r backend/requirements.txt` in the same Python environment used to start uvicorn, then restart the backend.",
        )
    except (jwt.PyJWTError, RuntimeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    try:
        return str(UUID(user_id))
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
            headers={"WWW-Authenticate": "Bearer"},
        )
