"""Interactive browser login for RunSignup OAuth (local callback server)."""

from __future__ import annotations

import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from race_predictor.data.runsignup_oauth import (
    OAuthClientConfig,
    OAuthTokens,
    build_authorize_url,
    exchange_authorization_code,
    generate_pkce_pair,
    save_tokens,
)


class RunSignupLoginError(Exception):
    """Raised when the browser login flow fails."""


def login_via_browser(config: OAuthClientConfig) -> OAuthTokens:
    """Run OAuth PKCE login and return tokens."""
    state = secrets.token_urlsafe(16)
    code_verifier, code_challenge = generate_pkce_pair()
    authorize_url = build_authorize_url(
        config,
        state=state,
        code_challenge=code_challenge,
    )

    parsed = urlparse(config.redirect_uri)
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise RunSignupLoginError(
            "CLI login requires a localhost redirect URI. "
            f"Set RUNSIGNUP_REDIRECT_URI (currently {config.redirect_uri!r})."
        )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    callback_path = parsed.path or "/"

    result: dict[str, str] = {}
    done = threading.Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path != callback_path:
                self.send_response(404)
                self.end_headers()
                return

            params = parse_qs(urlparse(self.path).query)
            result["code"] = (params.get("code") or [""])[0]
            result["state"] = (params.get("state") or [""])[0]
            result["error"] = (params.get("error") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>RunSignup login complete.</h2>"
                b"<p>You can close this tab and return to the terminal.</p></body></html>"
            )
            done.set()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer((parsed.hostname or "localhost", port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Opening browser for RunSignup login...\n{authorize_url}\n")
    webbrowser.open(authorize_url)

    if not done.wait(timeout=300):
        server.shutdown()
        raise RunSignupLoginError("Timed out waiting for OAuth callback (5 minutes).")

    server.shutdown()
    thread.join(timeout=1)

    if result.get("error"):
        raise RunSignupLoginError(f"RunSignup OAuth error: {result['error']}")
    if result.get("state") != state:
        raise RunSignupLoginError("OAuth state mismatch — possible CSRF attempt.")
    code = result.get("code", "").strip()
    if not code:
        raise RunSignupLoginError("OAuth callback did not include an authorization code.")

    return exchange_authorization_code(
        config,
        code=code,
        code_verifier=code_verifier,
    )


def login_and_save(config: OAuthClientConfig, token_path: str) -> OAuthTokens:
    tokens = login_via_browser(config)
    save_tokens(tokens, token_path)
    return tokens
