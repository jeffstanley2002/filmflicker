"""Supabase JWT verification. The frontend authenticates directly against
Supabase Auth (signup/login/session refresh) and sends the resulting JWT
as `Authorization: Bearer <token>` on every request here. No password ever
touches this backend - it only verifies a token Supabase already issued.
"""
import os
from pathlib import Path

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

_jwt_secret_cache = None
_jwk_client_cache = {}


def _jwt_secret() -> str:
    global _jwt_secret_cache
    if _jwt_secret_cache is not None:
        return _jwt_secret_cache
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET not set. Add it to backend/.env (see backend/.env.example).")
    _jwt_secret_cache = secret
    return secret


def _decode_with_jwks(token: str, alg: str) -> dict:
    """Verifies newer Supabase asymmetric JWTs using the project's JWKS.

    Supabase projects can issue access tokens signed by asymmetric keys
    instead of the legacy shared JWT secret. Those tokens advertise an
    algorithm such as ES256/RS256, so HS256-only verification rejects them
    with "The specified alg value is not allowed".
    """
    unverified = jwt.decode(token, options={"verify_signature": False})
    issuer = (unverified.get("iss") or os.environ.get("SUPABASE_URL") or "").rstrip("/")
    if not issuer:
        raise RuntimeError("SUPABASE_URL not set and token has no issuer claim.")
    if not issuer.endswith("/auth/v1"):
        issuer = f"{issuer}/auth/v1"

    jwks_url = f"{issuer}/.well-known/jwks.json"
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
            payload = jwt.decode(
                token,
                _jwt_secret(),
                algorithms=["HS256"],
                audience="authenticated",
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
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject claim")
    return user_id
