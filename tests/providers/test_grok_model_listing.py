"""Tests for the grok OAuth provider's graceful model-listing path.

grok OAuth bearers expire (~6h) and are renewed only by re-logging in. A 401
from the optional catalog endpoint therefore means the bearer expired — not
that the key is revoked — so ``list_model_ids`` must NOT permanently disable the
key (it has to stay usable for chat). On an all-key failure it falls back to
the last successfully persisted listing (disk cache) so the catalog keeps
populating grok models across restarts and bearer expiry.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from free_claude_code.config import grok_oauth
from free_claude_code.config.grok_oauth import (
    GROK_KNOWN_MODEL_IDS,
    load_cached_grok_model_ids,
    store_cached_grok_model_ids,
)
from free_claude_code.providers.base import ProviderConfig
from tests.providers.support import profiled_provider


def _grok_config(*, api_keys: tuple[str, ...]) -> ProviderConfig:
    return ProviderConfig(
        api_key=",".join(api_keys),
        api_keys=api_keys,
        base_url="https://cli-chat-proxy.grok.com/v1",
    )


def _grok_provider(api_keys: tuple[str, ...]):
    return profiled_provider("generic_openai", _grok_config(api_keys=api_keys))


def _fake_401_response() -> MagicMock:
    return MagicMock(status_code=401)


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    """Redirect the grok model cache to a tmp file (no ~/.fcc pollution)."""
    path = tmp_path / "grok-models.json"
    monkeypatch.setattr(grok_oauth, "grok_model_cache_path", lambda: path)
    return path


def _reattach_models_list(provider, staged) -> None:
    """Make every (re)built client's ``models.list`` resolve to ``staged``."""
    original_build = provider._build_client

    def _mocked_build():
        client = original_build()
        client.models.list = staged
        return client

    provider._build_client = _mocked_build
    provider._client.models.list = staged


class TestGrokModelListing:
    @pytest.mark.asyncio
    async def test_success_returns_ids_and_persists_cache(self, cache_path) -> None:
        provider = _grok_provider(("grok-bearer-A",))
        provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(id="grok-code"),
                    SimpleNamespace(id="grok-mini"),
                ]
            )
        )
        result = await provider.list_model_ids()
        assert result == frozenset({"grok-code", "grok-mini"})
        assert load_cached_grok_model_ids() == frozenset({"grok-code", "grok-mini"})
        assert cache_path.is_file()

    @pytest.mark.asyncio
    async def test_is_grok_provider_detected_by_base_url(self) -> None:
        assert _grok_provider(("grok-bearer-A",))._is_grok_provider is True

    @pytest.mark.asyncio
    async def test_401_does_not_permanently_disable_key(self, cache_path) -> None:
        provider = _grok_provider(("grok-bearer-A", "grok-bearer-B"))
        call_count = 0

        async def _two_stage() -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.AuthenticationError(
                    "expired bearer",
                    response=_fake_401_response(),
                    body=None,
                )
            return SimpleNamespace(data=[SimpleNamespace(id="grok-code")])

        _reattach_models_list(provider, _two_stage)

        result = await provider.list_model_ids()
        assert result == frozenset({"grok-code"})
        assert call_count == 2
        # Listing must NOT permanently disable either key.
        assert provider._key_pool._entries[0].permanently_failed is False
        assert provider._key_pool._entries[1].permanently_failed is False
        # The successful second probe still persisted the cache.
        assert load_cached_grok_model_ids() == frozenset({"grok-code"})

    @pytest.mark.asyncio
    async def test_non_auth_failure_also_degrades_without_disabling(
        self, cache_path
    ) -> None:
        provider = _grok_provider(("grok-bearer-A", "grok-bearer-B"))
        call_count = 0

        async def _flaky_then_ok() -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network unreachable")
            return SimpleNamespace(data=[SimpleNamespace(id="grok-code")])

        _reattach_models_list(provider, _flaky_then_ok)

        result = await provider.list_model_ids()
        assert result == frozenset({"grok-code"})
        assert call_count == 2
        assert provider._key_pool._entries[0].permanently_failed is False
        assert provider._key_pool._entries[1].permanently_failed is False

    @pytest.mark.asyncio
    async def test_all_keys_fail_without_cache_returns_empty_no_disable(
        self, cache_path
    ) -> None:
        provider = _grok_provider(("grok-bearer-A", "grok-bearer-B"))

        async def _always_fail() -> SimpleNamespace:
            raise openai.AuthenticationError(
                "expired",
                response=_fake_401_response(),
                body=None,
            )

        _reattach_models_list(provider, _always_fail)

        result = await provider.list_model_ids()
        # GROK_KNOWN_MODEL_IDS is empty by design; the disk cache (absent here)
        # is the authoritative fallback.
        assert result == frozenset(GROK_KNOWN_MODEL_IDS)
        assert provider._key_pool._entries[0].permanently_failed is False
        assert provider._key_pool._entries[1].permanently_failed is False

    @pytest.mark.asyncio
    async def test_all_keys_fail_loads_cached_fallback(self, cache_path) -> None:
        store_cached_grok_model_ids(frozenset({"grok-stale-but-known"}))
        provider = _grok_provider(("grok-bearer-A", "grok-bearer-B"))

        async def _always_fail() -> SimpleNamespace:
            raise openai.AuthenticationError(
                "expired",
                response=_fake_401_response(),
                body=None,
            )

        _reattach_models_list(provider, _always_fail)

        result = await provider.list_model_ids()
        assert result == frozenset({"grok-stale-but-known"})

    @pytest.mark.asyncio
    async def test_single_key_failure_falls_back_without_rotation(
        self, cache_path
    ) -> None:
        provider = _grok_provider(("grok-bearer-A",))

        async def _always_fail() -> SimpleNamespace:
            raise openai.AuthenticationError(
                "expired",
                response=_fake_401_response(),
                body=None,
            )

        _reattach_models_list(provider, _always_fail)

        result = await provider.list_model_ids()
        assert result == frozenset()
        assert provider._key_pool._entries[0].permanently_failed is False


class TestNonGrokProviderListingDivergence:
    """Non-grok providers keep the legacy posture: 401 permanently disables."""

    @pytest.mark.asyncio
    async def test_non_grok_401_permanently_disables_first_key(
        self, cache_path
    ) -> None:
        config = ProviderConfig(
            api_key=",".join(["sk-1", "sk-2"]),
            api_keys=("sk-1", "sk-2"),
            base_url="https://echo.example.com/v1",
        )
        provider = profiled_provider("generic_openai", config)
        assert provider._is_grok_provider is False

        call_count = 0

        async def _two_stage() -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.AuthenticationError(
                    "bad key",
                    response=_fake_401_response(),
                    body=None,
                )
            return SimpleNamespace(data=[SimpleNamespace(id="echo-model")])

        original_build = provider._build_client

        def _mocked_build():
            client = original_build()
            client.models.list = _two_stage
            return client

        provider._build_client = _mocked_build
        provider._client.models.list = _two_stage

        result = await provider.list_model_ids()
        assert result == frozenset({"echo-model"})
        assert call_count == 2
        # Non-grok: first key PERMANENTLY disabled; the grok disk cache untouched.
        assert provider._key_pool._entries[0].permanently_failed is True
        assert provider._key_pool._entries[1].permanently_failed is False
        assert not cache_path.is_file()
