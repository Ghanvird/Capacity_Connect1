# app_instance.py
from __future__ import annotations
from dash import Dash
import dash_bootstrap_components as dbc
import os, secrets
from pathlib import Path

def _load_or_create_secret_key() -> str:
    # 1) use env var if provided by ops
    env = os.environ.get("FLASK_SECRET_KEY")
    if env:
        return env.strip()

    # 2) persist a per-install key next to this file
    keyfile = Path(__file__).resolve().parent / ".flask_secret_key"
    try:
        if keyfile.exists():
            key = keyfile.read_text(encoding="utf-8").strip()
            if key:
                return key
        key = secrets.token_hex(32)  # ~256-bit
        keyfile.write_text(key, encoding="utf-8")
        try:
            os.chmod(keyfile, 0o600)  # best-effort on POSIX; ignored on Windows
        except Exception:
            pass
        return key
    except Exception:
        # fallback: ephemeral (sessions reset on restart)
        return secrets.token_hex(32)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="CAPACITY CONNECT",
)
server = app.server
server.config["SECRET_KEY"] = _load_or_create_secret_key()

# Optional hardening for production
if os.environ.get("FLASK_ENV") == "production":
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=True,  # requires HTTPS
    )
