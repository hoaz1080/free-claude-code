"""Tests for the grok OAuth account provisioning flow."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from free_claude_code.config import grok_oauth
from free_claude_code.config.custom_providers import (
    load_custom_providers_from_managed_env,
)
from free_claude_code.config.grok_oauth import (
    GROK_OAUTH_BASE_URL,
    GROK_OAUTH_PROVIDER_ID,
    GROK_OIDC_SCOPE,
    cancel_login,
    harvest_token_from_text,
    poll_login,
    start_device_auth_login,
    upsert_grok_oauth_account,
)


def _patch_env(tmp_path: Path, monkeypatch) -> Path:
    env_path = tmp_path / ".fcc" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # upsert goes through custom_providers, which binds managed_env_path at
    # import time — patch its reference, not paths.managed_env_path.
    monkeypatch.setattr(
        "free_claude_code.config.custom_providers.managed_env_path",
        lambda: env_path,
    )
    return env_path


def _write_grok_stub(tmp_path: Path, monkeypatch) -> Path:
    """Create an executable stub that mimics `grok login --device-auth`."""

    default_token = "grok-bearer-123"
    code = (
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys, time\n"
        "print('To sign in, open this URL in your browser:')\n"
        "print()\n"
        "print('  https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH')\n"
        "print()\n"
        "sys.stdout.flush()\n"
        "time.sleep(float(os.environ.get('GROK_STUB_SLEEP', '0.05')))\n"
        f"tok = os.environ.get('GROK_STUB_TOKEN', {default_token!r})\n"
        "home = os.environ['HOME']\n"
        "g = pathlib.Path(home) / '.grok'\n"
        "g.mkdir(parents=True, exist_ok=True)\n"
        f"(g / 'auth.json').write_text(json.dumps({{{GROK_OIDC_SCOPE!r}: {{'key': tok}}}}))\n"
    )
    stub = tmp_path / "grok_stub.py"
    stub.write_text(code, encoding="utf-8")
    os.chmod(stub, 0o755)
    monkeypatch.setenv("GROK_BIN", str(stub))
    return stub


def _clear_sessions() -> None:
    # Each test should start with a clean session store.
    grok_oauth._SESSIONS.clear()


class TestHarvestTokenFromText:
    def test_prefers_oidc_scope(self) -> None:
        text = json.dumps(
            {
                GROK_OIDC_SCOPE: {"key": "oidc-tok"},
                "https://accounts.x.ai/sign-in": {"key": "legacy-tok"},
            }
        )
        assert harvest_token_from_text(text) == "oidc-tok"

    def test_falls_back_to_legacy_scope(self) -> None:
        text = json.dumps({"https://accounts.x.ai/sign-in": {"key": "legacy-tok"}})
        assert harvest_token_from_text(text) == "legacy-tok"

    def test_falls_back_to_any_scope(self) -> None:
        text = json.dumps({"https://example.com/scope": {"key": "any-tok"}})
        assert harvest_token_from_text(text) == "any-tok"

    def test_strips_whitespace(self) -> None:
        text = json.dumps({GROK_OIDC_SCOPE: {"key": "  spaced-tok  "}})
        assert harvest_token_from_text(text) == "spaced-tok"

    def test_missing_key_returns_none(self) -> None:
        text = json.dumps({GROK_OIDC_SCOPE: {}})
        assert harvest_token_from_text(text) is None

    def test_non_string_key_returns_none(self) -> None:
        text = json.dumps({GROK_OIDC_SCOPE: {"key": 123}})
        assert harvest_token_from_text(text) is None

    def test_invalid_json_returns_none(self) -> None:
        assert harvest_token_from_text("{not json") is None

    def test_empty_returns_none(self) -> None:
        assert harvest_token_from_text("") is None
        assert harvest_token_from_text("   ") is None

    def test_non_object_returns_none(self) -> None:
        assert harvest_token_from_text("[1, 2, 3]") is None
        assert harvest_token_from_text("null") is None


class TestUpsertGrokOauthAccount:
    def test_creates_provider_when_absent(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        result = upsert_grok_oauth_account("tok-1")
        assert result == {"provider_id": GROK_OAUTH_PROVIDER_ID, "account_count": 1}

        providers = load_custom_providers_from_managed_env()
        defn = providers[GROK_OAUTH_PROVIDER_ID]
        assert defn.base_url == GROK_OAUTH_BASE_URL
        assert defn.api_keys == ("tok-1",)
        assert defn.display_name == "Grok (OAuth)"

    def test_appends_new_key_and_increments_count(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        upsert_grok_oauth_account("tok-1")
        result = upsert_grok_oauth_account("tok-2")
        assert result["account_count"] == 2

        defn = load_custom_providers_from_managed_env()[GROK_OAUTH_PROVIDER_ID]
        assert defn.api_keys == ("tok-1", "tok-2")

    def test_dedupes_existing_token(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        upsert_grok_oauth_account("tok-1")
        result = upsert_grok_oauth_account("tok-1")  # same account re-approved
        assert result["account_count"] == 1

        defn = load_custom_providers_from_managed_env()[GROK_OAUTH_PROVIDER_ID]
        assert defn.api_keys == ("tok-1",)

    def test_preserves_other_custom_providers(self, tmp_path, monkeypatch) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text(
            'FCC_CUSTOM_PROVIDERS="[{\\"provider_id\\":\\"other\\",'
            '\\"display_name\\":\\"Other\\",\\"base_url\\":'
            '\\"https://other.example/v1\\",\\"api_keys\\":[\\"k0\\"],'
            '\\"proxies\\":[],\\"detected_profile\\":null}]"\n'
        )
        upsert_grok_oauth_account("grok-tok")
        providers = load_custom_providers_from_managed_env()
        assert set(providers) == {"other", GROK_OAUTH_PROVIDER_ID}
        assert providers["other"].api_keys == ("k0",)

    def test_empty_token_raises(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            upsert_grok_oauth_account("   ")


class TestResolveGrokBin:
    def test_grok_bin_override_is_returned(self, monkeypatch, tmp_path) -> None:
        stub = tmp_path / "fake-grok"
        stub.write_text("#!/bin/sh\n")
        os.chmod(stub, 0o755)
        monkeypatch.setenv("GROK_BIN", str(stub))
        assert grok_oauth._resolve_grok_bin() == str(stub)


class TestDeviceAuthLoginFlow:
    @pytest.mark.asyncio
    async def test_start_returns_url_and_code(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        _write_grok_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_SLEEP", "10")  # keep alive for assertions

        result = await start_device_auth_login()
        assert "session_id" in result
        assert result["device_url"].startswith(
            "https://accounts.x.ai/oauth2/device?user_code="
        )
        assert result["user_code"] == "ABCD-EFGH"

        # Still pending (stub sleeping → no token, proc running).
        status = await poll_login(result["session_id"])
        assert status["status"] == "pending"

        cancel = await cancel_login(result["session_id"])
        assert cancel["ok"] is True
        assert grok_oauth._SESSIONS == {}

    @pytest.mark.asyncio
    async def test_poll_completes_with_token(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        _write_grok_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_SLEEP", "0.05")
        monkeypatch.setenv("GROK_STUB_TOKEN", "grok-real-tok")

        result = await start_device_auth_login()

        status = None
        for _ in range(50):
            status = await poll_login(result["session_id"])
            if status["status"] != "pending":
                break
            await asyncio.sleep(0.05)

        assert status["status"] == "complete"
        assert status["token"] == "grok-real-tok"
        assert grok_oauth._SESSIONS == {}  # completion cleans up

    @pytest.mark.asyncio
    async def test_cancel_cleans_up_session(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        _write_grok_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_SLEEP", "30")

        result = await start_device_auth_login()
        assert grok_oauth._SESSIONS  # a session exists

        cancel = await cancel_login(result["session_id"])
        assert cancel["status"] == "cancelled"

        # Polling a cancelled/unknown session is not_found, not pending.
        status = await poll_login(result["session_id"])
        assert status["status"] == "not_found"
        assert grok_oauth._SESSIONS == {}

    @pytest.mark.asyncio
    async def test_poll_unknown_session(self) -> None:
        _clear_sessions()
        status = await poll_login("does-not-exist")
        assert status["status"] == "not_found"
