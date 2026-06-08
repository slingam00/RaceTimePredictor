"""RunSignup OAuth2 authorization code flow with PKCE (CLI-friendly)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

RUNSIGNUP_BASE_URL = "https://runsignup.com"
DEFAULT_REDIRECT_URI = "http://localhost:8765/callback"
DEFAULT_TOKEN_PATH = Path("data/runsignup_tokens.json")
DEFAULT_SCOPE = "rsu_api_read"
AUTH_PATH = "/Profile/OAuth2/RequestGrant"
TOKEN_PATH = "/Rest/v2/auth/auth-code-redemption.json"


@dataclass(frozen=True)
class OAuthClientConfig:
    client_id: str
    client_secret: str
    redirect_uri: str


@dataclass
class OAuthTokens:
    access_token: str
    refresh_token: str | None
    expires_at: datetime | None
    scope: str | None = None
    token_type: str = "Bearer"


def oauth_config_from_env() -> OAuthClientConfig | None:
    client_id = os.getenv("RUNSIGNUP_CLIENT_ID", "").strip()
    client_secret = os.getenv("RUNSIGNUP_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    redirect_uri = os.getenv("RUNSIGNUP_REDIRECT_URI", DEFAULT_REDIRECT_URI).strip()
    return OAuthClientConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


def generate_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")
    return verifier, challenge


def build_authorize_url(
    config: OAuthClientConfig,
    *,
    state: str,
    code_challenge: str,
    scope: str = DEFAULT_SCOPE,
) -> str:
    params = {
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "response_type": "code",
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{RUNSIGNUP_BASE_URL}{AUTH_PATH}?{urllib.parse.urlencode(params)}"


def exchange_authorization_code(
    config: OAuthClientConfig,
    *,
    code: str,
    code_verifier: str,
) -> OAuthTokens:
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": code,
            "redirect_uri": config.redirect_uri,
            "code_verifier": code_verifier,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{RUNSIGNUP_BASE_URL}{TOKEN_PATH}",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return _parse_token_response(payload)


def load_tokens(path: str | Path = DEFAULT_TOKEN_PATH) -> OAuthTokens | None:
    token_path = Path(path)
    if not token_path.is_file():
        return None
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    expires_at = None
    if payload.get("expires_at"):
        expires_at = datetime.fromisoformat(payload["expires_at"])
    return OAuthTokens(
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        expires_at=expires_at,
        scope=payload.get("scope"),
        token_type=payload.get("token_type", "Bearer"),
    )


def save_tokens(tokens: OAuthTokens, path: str | Path = DEFAULT_TOKEN_PATH) -> Path:
    token_path = Path(path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at": tokens.expires_at.isoformat() if tokens.expires_at else None,
        "scope": tokens.scope,
        "token_type": tokens.token_type,
    }
    token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return token_path


def access_token_from_env_or_file(
    path: str | Path = DEFAULT_TOKEN_PATH,
) -> str | None:
    direct = os.getenv("RUNSIGNUP_ACCESS_TOKEN", "").strip()
    if direct:
        return direct
    tokens = load_tokens(path)
    if tokens is None:
        return None
    if tokens.expires_at is not None and tokens.expires_at <= datetime.now(timezone.utc):
        return None
    return tokens.access_token


def _parse_token_response(payload: dict[str, Any]) -> OAuthTokens:
    if "error" in payload:
        description = payload.get("error_description", payload["error"])
        raise RunSignupOAuthError(f"Token exchange failed: {description}")

    access_token = payload.get("access_token")
    if not access_token:
        raise RunSignupOAuthError("Token exchange response missing access_token")

    expires_in = payload.get("expires_in")
    expires_at = None
    if expires_in is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

    return OAuthTokens(
        access_token=str(access_token),
        refresh_token=payload.get("refresh_token"),
        expires_at=expires_at,
        scope=payload.get("scope"),
        token_type=payload.get("token_type", "Bearer"),
    )


class RunSignupOAuthError(Exception):
    """Raised when RunSignup OAuth fails."""
