"""Provider construction from declarative profiles and exceptional adapters."""

from collections.abc import Callable

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.dynamic_catalog import DynamicProviderCatalog
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.openai_chat import (
    GENERIC_OPENAI_PROFILE_ID,
    OPENAI_CHAT_PROFILES,
    create_openai_chat_provider,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .config import build_provider_config

ProviderFactory = Callable[
    [ProviderConfig, Settings, ProviderRateLimiter], BaseProvider
]


def _create_nvidia_nim(
    config: ProviderConfig,
    settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(
        config,
        nim_settings=settings.nim,
        rate_limiter=rate_limiter,
    )


def _create_open_router(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config, rate_limiter=rate_limiter)


def _create_mistral(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.mistral import MistralProvider

    return MistralProvider(config, rate_limiter=rate_limiter)


def _create_deepseek(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config, rate_limiter=rate_limiter)


def _create_lmstudio(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config, rate_limiter=rate_limiter)


def _create_cloudflare(
    config: ProviderConfig,
    settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.cloudflare import CloudflareProvider

    return CloudflareProvider(
        config,
        account_id=settings.cloudflare_account_id,
        rate_limiter=rate_limiter,
    )


def _create_gemini(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.gemini import GeminiProvider

    return GeminiProvider(config, rate_limiter=rate_limiter)


def _create_github_models(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.github_models import GitHubModelsProvider

    return GitHubModelsProvider(config, rate_limiter=rate_limiter)


_SPECIAL_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "mistral": _create_mistral,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "cloudflare": _create_cloudflare,
    "gemini": _create_gemini,
    "github_models": _create_github_models,
}

# Verify every static catalog entry has exactly one construction owner.
_static_ids = set(PROVIDER_CATALOG)
_profiled_ids = set(OPENAI_CHAT_PROFILES)
_special_ids = set(_SPECIAL_PROVIDER_FACTORIES)
_covered_static = _profiled_ids | _special_ids
if _profiled_ids & _special_ids:
    raise AssertionError(
        "Profiled and special provider IDs must not overlap: "
        f"overlap={_profiled_ids & _special_ids!r}"
    )
_missing = _static_ids - _covered_static
if _missing:
    raise AssertionError(
        f"Every static provider must have a construction owner: missing={_missing!r}"
    )


def _select_profile_id(
    provider_id: str,
    custom_def: CustomProviderDefinition | None,
    is_custom: bool,
) -> str:
    """Choose the appropriate OpenAI profile id for a provider.

    For static providers, the ``provider_id`` is the profile key.
    For custom providers, the ``detected_profile`` is used if known;
    otherwise the generic fallback.
    """
    if not is_custom:
        return provider_id
    if custom_def is not None and custom_def.detected_profile:
        detected = custom_def.detected_profile
        if detected in OPENAI_CHAT_PROFILES and detected != GENERIC_OPENAI_PROFILE_ID:
            return detected
    return GENERIC_OPENAI_PROFILE_ID


def create_provider(
    provider_id: str,
    settings: Settings,
    *,
    dynamic_catalog: DynamicProviderCatalog | None = None,
) -> BaseProvider:
    """Create a provider instance for a static or custom provider id.

    When *dynamic_catalog* is provided, custom providers are resolved
    from it. Custom providers use their ``detected_profile`` (or generic
    fallback) to select the appropriate ``OpenAIChatProfile``.
    """
    # Check static catalog first, then dynamic
    descriptor = PROVIDER_CATALOG.get(provider_id)
    is_custom = False
    custom_def: CustomProviderDefinition | None = None
    custom_api_keys: tuple[str, ...] | None = None

    if descriptor is None and dynamic_catalog is not None:
        custom_def = dynamic_catalog.get_custom_definition(provider_id)
        if custom_def is not None:
            descriptor = custom_def.to_descriptor()
            is_custom = True
            custom_api_keys = custom_def.api_keys

    if descriptor is None:
        if dynamic_catalog is not None:
            raise UnknownProviderError.for_provider(
                provider_id, dynamic_catalog.all_provider_ids
            )
        raise UnknownProviderError.for_provider(provider_id, PROVIDER_CATALOG)

    config = build_provider_config(
        descriptor,
        settings,
        custom_api_keys=custom_api_keys,
        custom_proxies=custom_def.proxies if custom_def and is_custom else None,
    )
    rate_limiter = ProviderRateLimiter(
        rate_limit=config.rate_limit or 40,
        rate_window=config.rate_window or 60.0,
        max_concurrency=config.max_concurrency,
    )

    # Custom providers never use special factories — only OpenAI profiles
    if not is_custom:
        factory = _SPECIAL_PROVIDER_FACTORIES.get(provider_id)
        if factory is not None:
            return factory(config, settings, rate_limiter)

    profile_id = _select_profile_id(provider_id, custom_def, is_custom)
    return create_openai_chat_provider(profile_id, config, rate_limiter)
