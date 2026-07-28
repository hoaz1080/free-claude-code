"""Per-provider default header injection (grok client headers, etc.).

The cli-chat-proxy server gate-checks ``x-grok-client-version``; the grok
OAuth provider stamps these headers on its ``CustomProviderDefinition`` and
the factory flattens them onto the constructed ``OpenAIChatProvider`` →
``AsyncOpenAI(default_headers=…)`` so every chat request carries them.
"""

from collections.abc import Mapping
from unittest.mock import patch

from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.grok_oauth import (
    GROK_OAUTH_DEFAULT_HEADERS,
    GROK_OAUTH_PROVIDER_ID,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    GENERIC_OPENAI_PROFILE_ID,
    create_openai_chat_provider,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from tests.providers.support import passthrough_rate_limiter


def _config() -> ProviderConfig:
    return ProviderConfig(
        api_key="grok-bearer",
        base_url="https://cli-chat-proxy.grok.com/v1",
        rate_limit=10,
        rate_window=60,
    )


def test_create_openai_chat_provider_forwards_default_headers():
    """Headers supplied to the factory reach the AsyncOpenAI client."""

    captured: dict[str, Mapping[str, str] | None] = {}

    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:

        def _ctor(*args, **kwargs):
            captured["default_headers"] = kwargs.get("default_headers")
            return mock_openai.return_value

        mock_openai.side_effect = _ctor
        create_openai_chat_provider(
            GENERIC_OPENAI_PROFILE_ID,
            _config(),
            passthrough_rate_limiter(),
            default_headers=dict(GROK_OAUTH_DEFAULT_HEADERS),
        )

    headers = captured["default_headers"]
    assert headers is not None
    assert headers["x-grok-client-version"] == GROK_OAUTH_DEFAULT_HEADERS[0][1]
    assert headers["x-grok-client-surface"] == GROK_OAUTH_DEFAULT_HEADERS[1][1]


def test_create_openai_chat_provider_no_headers_passes_none():
    """Without extra headers the factory passes None (no header injection)."""

    captured: dict[str, Mapping[str, str] | None] = {}

    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as mock_openai:

        def _ctor(*args, **kwargs):
            captured["default_headers"] = kwargs.get("default_headers")
            return mock_openai.return_value

        mock_openai.side_effect = _ctor
        create_openai_chat_provider(
            GENERIC_OPENAI_PROFILE_ID, _config(), passthrough_rate_limiter()
        )

    assert captured["default_headers"] is None


def test_create_openai_chat_provider_accepts_none_rate_limiter_unchanged():
    """Sanity: a None/default rate limiter path still constructs cleanly
    when default headers are provided (exercises the kw-only signature)."""

    limiter = ProviderRateLimiter(rate_limit=10, rate_window=60, max_concurrency=5)
    provider = create_openai_chat_provider(
        GENERIC_OPENAI_PROFILE_ID,
        _config(),
        limiter,
        default_headers=dict(GROK_OAUTH_DEFAULT_HEADERS),
    )
    # The provider stores the flattened headers ready to forward per request.
    assert provider._default_headers is not None
    assert dict(GROK_OAUTH_DEFAULT_HEADERS).items() <= provider._default_headers.items()


def test_grok_oauth_definition_default_headers_unaffected_by_other_fields():
    """A grok_oauth definition's default_headers are stable and tuple-typed."""

    defn = CustomProviderDefinition(
        provider_id=GROK_OAUTH_PROVIDER_ID,
        display_name="Grok (OAuth)",
        base_url="https://cli-chat-proxy.grok.com/v1",
        api_keys=("k0", "k1"),
        proxies=(),
        detected_profile=None,
        default_headers=GROK_OAUTH_DEFAULT_HEADERS,
    )
    assert defn.default_headers == GROK_OAUTH_DEFAULT_HEADERS
    assert all(isinstance(p, tuple) and len(p) == 2 for p in defn.default_headers)
