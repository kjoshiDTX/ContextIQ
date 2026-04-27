"""Supabase JWT authentication dependency for FastAPI endpoints."""

from typing import Annotated
from fastapi import Depends, HTTPException, Header
import jwt
from jwt import PyJWKClient

from app.core.config import get_settings

# Cached JWKS client — fetches public keys from Supabase once and caches them
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        settings = get_settings()
        jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True)
    return _jwks_client


async def get_current_user(authorization: str = Header(...)) -> str:
    """Verify the Supabase JWT from the Authorization header.

    Expects: Authorization: Bearer <supabase_access_token>
    Returns the authenticated user's UUID — matches auth.users(id).
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = authorization[7:]

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256", "HS256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired — please sign in again")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth error: {e}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject claim")

    return user_id


# Shorthand type alias — use this in endpoint signatures:
#   async def my_endpoint(user_id: CurrentUser, ...):
CurrentUser = Annotated[str, Depends(get_current_user)]


async def get_optional_user(authorization: str = Header(default=None)) -> str | None:
    """Like get_current_user but returns None instead of 401 when no token is provided.

    Use for endpoints that work both authenticated (personalized) and unauthenticated (public).
    """
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except Exception:
        return None


OptionalUser = Annotated[str | None, Depends(get_optional_user)]
