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
    GROK_OAUTH_DEFAULT_HEADERS,
    GROK_OAUTH_PROVIDER_ID,
    GROK_OIDC_SCOPE,
    cancel_login,
    harvest_token_from_text,
    mask_proxy_url,
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
    # The proxy pool reads paths.managed_env_path fresh each call — redirect it
    # to the same tmp file so login-side proxy rotation is hermetic and reads
    # whatever FCC_PROXY_POOL line the test seeds into env_path.
    monkeypatch.setattr(
        "free_claude_code.config.paths.managed_env_path",
        lambda: env_path,
    )
    return env_path


def _seed_proxy_pool(env_path: Path, urls: tuple[str, ...]) -> None:
    """Write a FCC_PROXY_POOL line of healthy proxies into the managed env."""
    import json as _json

    entries = [
        {"url": u, "label": "", "healthy": True, "last_tested": 0.0} for u in urls
    ]
    value = _json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    line = f'FCC_PROXY_POOL="{escaped}"\n'
    existing = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    if "FCC_PROXY_POOL=" in existing:
        lines = existing.splitlines()
        for i, existing_line in enumerate(lines):
            if existing_line.strip().startswith("FCC_PROXY_POOL="):
                lines[i] = line.rstrip("\n")
                break
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text((existing if existing else "") + line, encoding="utf-8")


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


class TestGrokClientHeaders:
    def test_default_headers_match_official_cli(self) -> None:
        names = {h[0] for h in GROK_OAUTH_DEFAULT_HEADERS}
        assert names == {"x-grok-client-version", "x-grok-client-surface"}

    def test_upsert_stamps_headers_on_create(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        upsert_grok_oauth_account("tok-1")
        defn = load_custom_providers_from_managed_env()[GROK_OAUTH_PROVIDER_ID]
        assert defn.default_headers == GROK_OAUTH_DEFAULT_HEADERS

    def test_upsert_preserves_headers_on_re_add(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        upsert_grok_oauth_account("tok-1")
        upsert_grok_oauth_account("tok-1")  # duplicate → no-op re-approval
        defn = load_custom_providers_from_managed_env()[GROK_OAUTH_PROVIDER_ID]
        assert defn.default_headers == GROK_OAUTH_DEFAULT_HEADERS
        assert defn.api_keys == ("tok-1",)

    def test_upsert_stamps_headers_on_append(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        upsert_grok_oauth_account("tok-1")
        upsert_grok_oauth_account("tok-2")
        defn = load_custom_providers_from_managed_env()[GROK_OAUTH_PROVIDER_ID]
        assert defn.api_keys == ("tok-1", "tok-2")
        assert defn.default_headers == GROK_OAUTH_DEFAULT_HEADERS

    def test_backfill_headers_for_legacy_definition_without_them(
        self, tmp_path, monkeypatch
    ) -> None:
        # A grok_oauth entry persisted before headers existed has empty headers
        # in the file, but the dynamic catalog must serve the grok headers so
        # pre-existing accounts survive the 426 gate without re-adding.
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text(
            'FCC_CUSTOM_PROVIDERS="[{\\"provider_id\\":\\"grok_oauth\\",'
            '\\"display_name\\":\\"Grok (OAuth)\\",'
            '\\"base_url\\":\\"https://cli-chat-proxy.grok.com/v1\\",'
            '\\"api_keys\\":[\\"legacy-tok\\"],\\"proxies\\":[],'
            '\\"detected_profile\\":null}]"\n'
        )
        from free_claude_code.config.dynamic_catalog import DynamicProviderCatalog

        catalog = DynamicProviderCatalog()
        defn = catalog.get_custom_definition(GROK_OAUTH_PROVIDER_ID)
        assert defn is not None
        assert defn.default_headers == GROK_OAUTH_DEFAULT_HEADERS
        assert defn.api_keys == ("legacy-tok",)


class TestMaskProxyUrl:
    def test_masks_password(self) -> None:
        assert mask_proxy_url("http://user:secret@host:8080") == "http://****@host:8080"

    def test_no_credentials_unchanged(self) -> None:
        assert mask_proxy_url("http://host:8080") == "http://host:8080"

    def test_socks5_masks(self) -> None:
        assert (
            mask_proxy_url("socks5://u:p@1.2.3.4:1080") == "socks5://****@1.2.3.4:1080"
        )

    def test_empty_unchanged(self) -> None:
        assert mask_proxy_url("") == ""


class TestLoginProxySelection:
    def test_empty_pool_returns_empty(self, tmp_path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        assert grok_oauth._select_login_proxy() == ""

    def test_picks_proxy_zero_for_first_account(self, tmp_path, monkeypatch) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        _seed_proxy_pool(env_path, ("http://a:8080", "http://b:8080", "http://c:8080"))
        assert grok_oauth._select_login_proxy() == "http://a:8080"

    def test_rotates_per_account_and_wraps(self, tmp_path, monkeypatch) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        _seed_proxy_pool(env_path, ("http://a:8080", "http://b:8080", "http://c:8080"))
        # Account 0 (none yet) -> proxy[0]
        assert grok_oauth._select_login_proxy() == "http://a:8080"
        upsert_grok_oauth_account("k0")
        assert grok_oauth._select_login_proxy() == "http://b:8080"  # account 1
        upsert_grok_oauth_account("k1")
        assert grok_oauth._select_login_proxy() == "http://c:8080"  # account 2
        upsert_grok_oauth_account("k2")
        assert grok_oauth._select_login_proxy() == "http://a:8080"  # wraps to % len


class TestLoginProxyEnvInjection:
    @pytest.mark.asyncio
    async def test_login_injects_pool_proxy_env(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        env_path = _patch_env(tmp_path, monkeypatch)
        _seed_proxy_pool(env_path, ("http://a:8080", "http://b:8080"))
        _write_grok_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_SLEEP", "30")  # keep alive for assertions

        captured: list[dict | None] = []
        orig_exec = asyncio.create_subprocess_exec

        async def spy_exec(*args, **kwargs):
            captured.append(kwargs.get("env"))
            return await orig_exec(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)

        await start_device_auth_login()

        # No grok accounts yet -> login uses proxy[0].
        assert captured
        login_env = captured[-1]
        assert login_env is not None
        assert login_env["HTTPS_PROXY"] == "http://a:8080"
        assert login_env["HTTP_PROXY"] == "http://a:8080"
        assert login_env["ALL_PROXY"] == "http://a:8080"
        # Clean up the running child so the temp home is released.
        for s in list(grok_oauth._SESSIONS.values()):
            grok_oauth._cleanup_session(s)
        assert grok_oauth._SESSIONS == {}

    @pytest.mark.asyncio
    async def test_login_without_pool_going_direct(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        _patch_env(tmp_path, monkeypatch)  # no FCC_PROXY_POOL seeded
        _write_grok_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_SLEEP", "30")

        captured: list[dict | None] = []
        orig_exec = asyncio.create_subprocess_exec

        async def spy_exec(*args, **kwargs):
            captured.append(kwargs.get("env"))
            return await orig_exec(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)

        await start_device_auth_login()

        assert captured
        login_env = captured[-1]
        assert login_env is not None
        # No pool proxy injected: only HOME differs from the process env.
        assert grok_oauth._select_login_proxy() == ""
        for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            assert login_env.get(var, os.environ.get(var)) == os.environ.get(var)
        for s in list(grok_oauth._SESSIONS.values()):
            grok_oauth._cleanup_session(s)
        assert grok_oauth._SESSIONS == {}


def _write_grok_retry_stub(tmp_path: Path, monkeypatch) -> Path:
    """Stub that fails (no URL) when given a "dead" proxy, succeeds otherwise.

    Mirrors a real grok login whose selected proxy can't tunnel: it exits
    nonzero without printing a device URL. ``GROK_STUB_DEAD`` (comma list)
    names the proxy URLs (or ``""`` for the direct attempt) that must fail.
    Any candidate not in the dead list prints the device URL and stays alive.
    """
    code = (
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "dead = set(p.strip() for p in os.environ.get('GROK_STUB_DEAD','').split(','))\n"
        "proxy = (os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') "
        "or os.environ.get('ALL_PROXY') or '')\n"
        "key = proxy if proxy else 'direct'\n"
        "if key in dead:\n"
        "    sys.exit(1)\n"  # simulate the 502 cause: no device URL
        "print('To sign in, open this URL in your browser:')\n"
        "print()\n"
        "print('  https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH')\n"
        "print()\n"
        "sys.stdout.flush()\n"
        "time.sleep(float(os.environ.get('GROK_STUB_SLEEP','30')))\n"
    )
    stub = tmp_path / "grok_retry_stub.py"
    stub.write_text(code, encoding="utf-8")
    os.chmod(stub, 0o755)
    monkeypatch.setenv("GROK_BIN", str(stub))
    return stub


def _captured_proxies(captured: list[dict | None]) -> list[str]:
    """Extract the per-attempt assigned proxy URL (or '' for direct)."""
    out: list[str] = []
    for env in captured:
        if env is None:
            out.append("")
            continue
        out.append(
            env.get("HTTPS_PROXY")
            or env.get("HTTP_PROXY")
            or env.get("ALL_PROXY")
            or ""
        )
    return out


class TestLoginProxyRetry:
    """start_device_auth_login rotates candidates instead of 502-ing on a bad proxy."""

    @pytest.mark.asyncio
    async def test_dead_first_proxy_rotates_to_next(
        self, tmp_path, monkeypatch
    ) -> None:
        _clear_sessions()
        env_path = _patch_env(tmp_path, monkeypatch)
        _seed_proxy_pool(env_path, ("http://a:8080", "http://b:8080"))
        _write_grok_retry_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_DEAD", "http://a:8080")
        monkeypatch.setenv("GROK_STUB_SLEEP", "30")

        captured: list[dict | None] = []
        orig_exec = asyncio.create_subprocess_exec

        async def spy_exec(*args, **kwargs):
            captured.append(kwargs.get("env"))
            return await orig_exec(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)

        result = await start_device_auth_login()
        assert result["device_url"].startswith("https://accounts.x.ai/oauth2/device")

        # Tried the dead proxy a first, then succeeded via b.
        tried = _captured_proxies(captured)
        assert tried[0] == "http://a:8080"
        assert tried[-1] == "http://b:8080"
        assert len(tried) == 2
        for s in list(grok_oauth._SESSIONS.values()):
            grok_oauth._cleanup_session(s)
        assert grok_oauth._SESSIONS == {}

    @pytest.mark.asyncio
    async def test_all_pool_proxies_dead_falls_back_to_direct(
        self, tmp_path, monkeypatch
    ) -> None:
        _clear_sessions()
        env_path = _patch_env(tmp_path, monkeypatch)
        _seed_proxy_pool(env_path, ("http://a:8080", "http://b:8080"))
        _write_grok_retry_stub(tmp_path, monkeypatch)
        monkeypatch.setenv("GROK_STUB_DEAD", "http://a:8080,http://b:8080")
        monkeypatch.setenv("GROK_STUB_SLEEP", "30")

        captured: list[dict | None] = []
        orig_exec = asyncio.create_subprocess_exec

        async def spy_exec(*args, **kwargs):
            captured.append(kwargs.get("env"))
            return await orig_exec(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)

        result = await start_device_auth_login()
        assert result["device_url"].startswith("https://accounts.x.ai/oauth2/device")

        tried = _captured_proxies(captured)
        assert tried[0] == "http://a:8080"
        assert tried[1] == "http://b:8080"
        assert tried[-1] == ""  # direct fallback succeeded
        for s in list(grok_oauth._SESSIONS.values()):
            grok_oauth._cleanup_session(s)
        assert grok_oauth._SESSIONS == {}

    @pytest.mark.asyncio
    async def test_every_candidate_fails_raises(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        env_path = _patch_env(tmp_path, monkeypatch)
        _seed_proxy_pool(env_path, ("http://a:8080",))
        _write_grok_retry_stub(tmp_path, monkeypatch)
        # Both the pool proxy and the direct fallback are dead.
        monkeypatch.setenv("GROK_STUB_DEAD", "http://a:8080,direct")

        captured: list[dict | None] = []
        orig_exec = asyncio.create_subprocess_exec

        async def spy_exec(*args, **kwargs):
            captured.append(kwargs.get("env"))
            return await orig_exec(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)

        with pytest.raises(RuntimeError, match="device verification URL"):
            await start_device_auth_login()
        # No session leaked.
        assert grok_oauth._SESSIONS == {}
        tried = _captured_proxies(captured)
        assert "http://a:8080" in tried and "" in tried

    @pytest.mark.asyncio
    async def test_empty_pool_succeeds_directly(self, tmp_path, monkeypatch) -> None:
        _clear_sessions()
        _patch_env(tmp_path, monkeypatch)  # no pool
        _write_grok_retry_stub(tmp_path, monkeypatch)  # nothing in DEAD list
        monkeypatch.setenv("GROK_STUB_SLEEP", "30")

        result = await start_device_auth_login()
        assert result["device_url"].startswith("https://accounts.x.ai/oauth2/device")
        for s in list(grok_oauth._SESSIONS.values()):
            grok_oauth._cleanup_session(s)
        assert grok_oauth._SESSIONS == {}
