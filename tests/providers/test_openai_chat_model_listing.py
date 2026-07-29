"""Tests for ``list_model_ids`` retry across API keys."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from free_claude_code.providers.base import ProviderConfig
from tests.providers.support import profiled_provider


def _echoing_provider_config(*, api_keys: list[str]) -> ProviderConfig:
    """Return a config with the given list of API keys."""
    return ProviderConfig(
        api_key=",".join(api_keys),
        api_keys=tuple(api_keys),
        base_url="https://echo.example.com/v1",
    )


# Canned 401 response for AuthenticationError — annotated as ``MagicMock`` so
# ty accepts it when passed to ``openai.AuthenticationError(response=...)``.
def _fake_401_response() -> MagicMock:
    """Return a minimal httpx.Response stub for constructing auth errors."""
    return MagicMock(status_code=401)


@pytest.fixture
def echo_provider_two_keys():
    """Provider with two API keys for retry tests."""
    return profiled_provider(
        "minimax",
        _echoing_provider_config(
            api_keys=["sk-key-aaaa", "sk-key-bbbb"],
        ),
    )


class TestListModelIdsRetry:
    """``list_model_ids`` retries across keys on authentication failures."""

    @pytest.mark.asyncio
    async def test_single_attempt_passes_when_key_is_valid(
        self, echo_provider_two_keys
    ):
        """Single attempt succeeds when the first key is accepted."""
        echo_provider_two_keys._client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(id="model-A"), SimpleNamespace(id="model-B")],
            )
        )
        result = await echo_provider_two_keys.list_model_ids()
        assert result == frozenset({"model-A", "model-B"})

    @pytest.mark.asyncio
    async def test_retries_on_auth_error_with_next_key(
        self,
        echo_provider_two_keys,
    ):
        """First key fails with AuthenticationError → rotate → second key succeeds."""
        call_count = 0

        async def _two_stage_list() -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.AuthenticationError(
                    "invalid token",
                    response=_fake_401_response(),
                    body=None,
                )
            return SimpleNamespace(
                data=[SimpleNamespace(id="model-from-key-2")],
            )

        # Mock _build_client so that the rebuilt client (after rotation)
        # still carries our controlled ``models.list``.
        original_build = echo_provider_two_keys._build_client

        def _mocked_build():
            client = original_build()
            client.models.list = _two_stage_list
            return client

        echo_provider_two_keys._build_client = _mocked_build
        echo_provider_two_keys._client.models.list = _two_stage_list

        result = await echo_provider_two_keys.list_model_ids()
        assert result == frozenset({"model-from-key-2"})
        assert call_count == 2

        # First key should now be marked as permanently failed
        assert echo_provider_two_keys._key_pool._entries[0].permanently_failed is True
        # Second key should still be available
        assert echo_provider_two_keys._key_pool._entries[1].permanently_failed is False

    @pytest.mark.asyncio
    async def test_raises_auth_error_when_all_keys_exhausted(
        self,
        echo_provider_two_keys,
    ):
        """When every key returns AuthenticationError, raise to caller."""
        auth_error = openai.AuthenticationError(
            "Auth fail for both keys",
            response=_fake_401_response(),
            body=None,
        )

        # Wrap in a side-effect callable so each invocation raises a fresh error
        call_count = 0

        async def _always_fail() -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            raise auth_error

        # Mock _build_client so rebuilt clients also fail.
        original_build = echo_provider_two_keys._build_client

        def _mocked_build():
            client = original_build()
            client.models.list = _always_fail
            return client

        echo_provider_two_keys._build_client = _mocked_build
        echo_provider_two_keys._client.models.list = _always_fail

        with pytest.raises(openai.AuthenticationError):
            await echo_provider_two_keys.list_model_ids()

        # Both keys were tried, both marked permanently failed
        assert call_count == 2
        assert echo_provider_two_keys._key_pool._entries[0].permanently_failed is True
        assert echo_provider_two_keys._key_pool._entries[1].permanently_failed is True

    @pytest.mark.asyncio
    async def test_non_auth_errors_raise_immediately(
        self,
        echo_provider_two_keys,
    ):
        """Non-authentication errors (e.g. network) are raised without retry."""
        echo_provider_two_keys._client.models.list = AsyncMock(
            side_effect=ConnectionError("Network unreachable"),
        )

        with pytest.raises(ConnectionError):
            await echo_provider_two_keys.list_model_ids()

        # No keys should be marked permanently failed — this was not an auth error
        assert echo_provider_two_keys._key_pool._entries[0].permanently_failed is False
        assert echo_provider_two_keys._key_pool._entries[1].permanently_failed is False
