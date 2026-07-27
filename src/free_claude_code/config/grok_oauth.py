"""Grok OAuth account provisioning via the official ``grok`` CLI.

The Grok Code CLI authenticates with a device-code OAuth flow against
``accounts.x.ai`` (``auth.x.ai`` issuer). Rather than reimplement that
handshake — which would require hard-coding xAI's ``client_id`` and endpoints
and would break whenever xAI rotates them — we drive the official
``grok login --device-auth`` binary as a subprocess.

Each login runs under an **isolated ``HOME``** (a fresh temp dir) so that
multiple accounts never clobber one ``~/.grok/auth.json``. On success the CLI
writes ``~/.grok/auth.json`` shaped ``{"<scope>": {"key": "<bearer>"}}``; we
harvest that bearer and store it as one key of the ``grok_oauth`` custom
provider (targeting the CLI's chat proxy ``cli-chat-proxy.grok.com/v1``).
The existing ``ApiKeyPool`` then rotates across accounts on rate limits.

This module lives in the config package and must not import from
``free_claude_code.providers`` (the import-boundary contract test enforces
that the config package stays provider-free).
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from free_claude_code.config.custom_providers import (
    CustomProviderDefinition,
    load_custom_providers_from_managed_env,
    save_custom_providers_to_managed_env,
)

GROK_OAUTH_PROVIDER_ID = "grok_oauth"
GROK_OAUTH_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
# OIDC scope baked into the grok installer; the harvested ``key`` under this
# scope is the bearer used against the chat proxy.
GROK_OIDC_SCOPE = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"

# Sessions auto-expire if not touched within this many seconds (user walked
# away without approving in the browser).
_SESSION_TTL_SECONDS = 600

# Override the path to the grok binary (set ``GROK_BIN`` in the environment,
# mainly for tests using a stub). When unset we resolve the real install.
_GROK_BIN_ENV = "GROK_BIN"


def _resolve_grok_bin() -> str:
    """Return the absolute path to the ``grok`` CLI binary.

    Resolved against the **real** ``HOME`` (never the isolated login ``HOME``),
    so an isolated subprocess can still find the installed binary. Honours a
    ``GROK_BIN`` override (tests use a stub). Raises ``FileNotFoundError`` if
    no binary is available.
    """

    override = os.environ.get(_GROK_BIN_ENV)
    if override:
        return override
    candidates = [
        os.path.join(os.path.expanduser("~"), ".grok", "bin", "grok"),
        shutil.which("grok") or "",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "The 'grok' CLI was not found. Install it with: "
        "curl -fsSL https://x.ai/cli/install.sh | bash"
    )


_DEVICE_URL_RE = re.compile(r"https?://[\w.-]+/oauth2/device\?\S*user_code=\S+")
# A user code is printed on its own line, e.g. ``DFDH-SCJX``.
_USER_CODE_RE = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{4})\b")


@dataclass(slots=True)
class GrokLoginSession:
    """One in-flight device-code login."""

    session_id: str
    work_home: Path
    proc: asyncio.subprocess.Process
    device_url: str
    user_code: str
    created_at: float = field(default_factory=lambda: time.monotonic())


_SESSIONS: dict[str, GrokLoginSession] = {}


def _new_work_home() -> Path:
    return Path(tempfile.mkdtemp(prefix="grok-login-"))


def _cleanup_session(session: GrokLoginSession) -> None:
    _proc_kill(session.proc)
    shutil.rmtree(session.work_home, ignore_errors=True)
    _SESSIONS.pop(session.session_id, None)


def _proc_kill(proc: asyncio.subprocess.Process) -> None:
    try:
        if proc.returncode is None:
            proc.kill()
    except ProcessLookupError:
        pass


# Per-line read deadline while waiting for the CLI to print its URL/code. The
# CLI emits these within the first second and then blocks polling for approval;
# a generous bound lets us react if a future version prints them slowly
# without hanging the request forever if it never emits them.
_READ_TIMEOUT_SECONDS = 15.0


async def _read_login_output(proc: asyncio.subprocess.Process) -> tuple[str, str]:
    """Read enough of ``grok login`` stdout to capture URL + code.

    The CLI prints the verification URL and user code quickly, then blocks
    while polling for approval. We stop as soon as we have both so the caller
    can hand them to the UI while the child keeps polling. Since the user code
    is embedded in the verification URL (``user_code=XXXX-XXXX``), both are
    captured on the URL line.
    """

    stdout = proc.stdout
    assert stdout is not None
    device_url = ""
    user_code = ""
    while not (device_url and user_code):
        try:
            raw = await asyncio.wait_for(
                stdout.readline(), timeout=_READ_TIMEOUT_SECONDS
            )
        except TimeoutError, asyncio.IncompleteReadError:
            break
        if not raw:
            break
        text = raw.decode("utf-8", errors="replace").rstrip()
        if not device_url and (match := _DEVICE_URL_RE.search(text)):
            device_url = match.group(0)
        if not user_code and (match := _USER_CODE_RE.search(text)):
            user_code = match.group(1)
    return device_url, user_code


async def start_device_auth_login() -> dict[str, str]:
    """Start a ``grok login --device-auth`` flow and return its device URL/code.

    Returns ``{"session_id", "device_url", "user_code"}``. Raises
    ``FileNotFoundError`` if grok isn't installed or ``RuntimeError`` if the
    CLI failed to produce a device URL/user code.
    """

    grok_bin = _resolve_grok_bin()
    work_home = _new_work_home()
    session_id = uuid.uuid4().hex
    env = {**os.environ, "HOME": str(work_home)}

    proc = await asyncio.create_subprocess_exec(
        grok_bin,
        "login",
        "--device-auth",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    session = GrokLoginSession(
        session_id=session_id,
        work_home=work_home,
        proc=proc,
        device_url="",
        user_code="",
    )
    _SESSIONS[session_id] = session

    try:
        device_url, user_code = await _read_login_output(proc)
    except Exception:
        _cleanup_session(session)
        raise

    session.device_url = device_url
    session.user_code = user_code

    if not device_url:
        _cleanup_session(session)
        raise RuntimeError(
            "grok login did not produce a device verification URL. "
            "Check that the grok CLI is installed and up to date."
        )

    return {
        "session_id": session_id,
        "device_url": device_url,
        "user_code": user_code,
    }


def harvest_token_from_text(auth_json_text: str) -> str | None:
    """Extract a bearer token from a ``grok`` auth.json document, preferring
    the OIDC scope. Returns ``None`` for malformed/empty input.

    Shape: ``{"<scope>": {"key": "<token>", ...}, ...}`` — the groove used by
    the ``read_grok_token`` helper in the official installer.
    """

    if not auth_json_text:
        return None
    try:
        parsed = json.loads(auth_json_text)
    except json.JSONDecodeError, TypeError:
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None

    def _key_for(scope: str) -> str | None:
        entry = parsed.get(scope)
        if isinstance(entry, dict):
            key = entry.get("key")
            if isinstance(key, str) and key.strip():
                return key.strip()
        return None

    token = _key_for(GROK_OIDC_SCOPE)
    if token:
        return token
    # Fall back to the legacy scope, then any scope present.
    for entry in parsed.values():
        if isinstance(entry, dict):
            key = entry.get("key")
            if isinstance(key, str) and key.strip():
                return key.strip()
    return None


def harvest_token(work_home: Path) -> str | None:
    """Read the harvested bearer from ``<work_home>/.grok/auth.json``."""

    auth_path = work_home / ".grok" / "auth.json"
    try:
        text = auth_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return harvest_token_from_text(text)


def _expired(session: GrokLoginSession, now: float | None = None) -> bool:
    if now is None:
        now = time.monotonic()
    return (now - session.created_at) > _SESSION_TTL_SECONDS


async def poll_login(session_id: str) -> dict[str, str]:
    """Poll an in-flight login. On completion (token harvested) the session is
    cleaned up and ``status == "complete"`` with the bearer. On the child
    exiting without a token, ``status == "error"``. Still-waiting flows report
    ``status == "pending"``. Expired sessions are cleaned up and reported as
    ``status == "expired"``.
    """

    session = _SESSIONS.get(session_id)
    if session is None:
        return {"status": "not_found"}
    if _expired(session):
        _cleanup_session(session)
        return {"status": "expired"}

    token = harvest_token(session.work_home)
    if token:
        _cleanup_session(session)
        return {"status": "complete", "token": token}

    if session.proc.returncode is not None:
        # Child finished without writing a harvestable token (denied, network
        # failure, timeout). Drain nothing more — surface as error.
        _cleanup_session(session)
        return {"status": "error", "error": "grok login exited without a token"}

    return {"status": "pending"}


async def cancel_login(session_id: str) -> dict[str, bool | str]:
    """Cancel and clean up an in-flight login."""

    session = _SESSIONS.pop(session_id, None)
    if session is None:
        return {"status": "not_found"}
    _cleanup_session(session)
    return {"status": "cancelled", "ok": True}


def upsert_grok_oauth_account(token: str) -> dict[str, str | int]:
    """Append a harvested grok OAuth bearer to the ``grok_oauth`` custom
    provider (creating it if absent), deduped. Persists to the managed env.

    Returns ``{"provider_id": "grok_oauth", "account_count": N}``.
    """

    token = token.strip()
    if not token:
        raise ValueError("token must be a non-empty string")

    definitions = load_custom_providers_from_managed_env()
    existing = definitions.get(GROK_OAUTH_PROVIDER_ID)
    if existing is None:
        definition = CustomProviderDefinition(
            provider_id=GROK_OAUTH_PROVIDER_ID,
            display_name="Grok (OAuth)",
            base_url=GROK_OAUTH_BASE_URL,
            api_keys=(token,),
            proxies=(),
            detected_profile=None,
        )
        definitions[GROK_OAUTH_PROVIDER_ID] = definition
    else:
        if token in existing.api_keys:
            logger.info("Grok OAuth token already present; not re-adding")
        else:
            definitions[GROK_OAUTH_PROVIDER_ID] = CustomProviderDefinition(
                provider_id=existing.provider_id,
                display_name=existing.display_name or "Grok (OAuth)",
                base_url=existing.base_url or GROK_OAUTH_BASE_URL,
                api_keys=(*existing.api_keys, token),
                proxies=existing.proxies,
                detected_profile=existing.detected_profile,
            )
    save_custom_providers_to_managed_env(definitions)

    account_count = len(definitions[GROK_OAUTH_PROVIDER_ID].api_keys)
    logger.info("Grok OAuth account provisioned ({})", account_count)
    return {"provider_id": GROK_OAUTH_PROVIDER_ID, "account_count": account_count}
