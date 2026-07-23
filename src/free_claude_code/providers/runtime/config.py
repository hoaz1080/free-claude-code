"""Provider configuration construction from neutral catalog metadata."""

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.custom_providers import merge_legacy_keys, parse_api_keys
from free_claude_code.config.provider_catalog import ProviderDescriptor
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig


def string_setting(settings: Settings, attr_name: str | None, default: str = "") -> str:
    """Return a string-valued settings attribute, ignoring non-string mocks."""
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def provider_credential(descriptor: ProviderDescriptor, settings: Settings) -> str:
    """Return the configured credential for a provider descriptor."""
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    if descriptor.credential_attr:
        return string_setting(settings, descriptor.credential_attr)
    return ""


def require_provider_credential(
    descriptor: ProviderDescriptor, credential: str
) -> None:
    """Raise a user-facing configuration error when a required key is missing."""
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    message = f"{descriptor.credential_env} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise ApplicationUnavailableError(message)


def build_provider_config(
    descriptor: ProviderDescriptor,
    settings: Settings,
    *,
    custom_api_keys: tuple[str, ...] | None = None,
    custom_proxies: tuple[str, ...] | None = None,
) -> ProviderConfig:
    """Build shared provider configuration for one provider descriptor.

    When *custom_api_keys* is provided, those keys are used directly
    (for custom providers whose keys don't come from env vars).
    Otherwise, the credential from *descriptor* is parsed for
    comma-separated keys (backward-compatible multi-key support).

    When *custom_proxies* is provided, those are aligned with the keys.
    For static providers, the proxy env var is also parsed for
    comma-separated values aligned with keys.
    """
    if custom_api_keys is not None:
        api_keys = custom_api_keys
        api_key = api_keys[0] if api_keys else ""
        proxies = custom_proxies or ()
    else:
        credential = provider_credential(descriptor, settings)
        if descriptor.credential_env is not None:
            require_provider_credential(descriptor, credential)
        api_keys = merge_legacy_keys(descriptor, credential)
        api_key = credential.strip() if credential else ""
        # Parse proxy env var into per-key proxies (comma-separated)
        proxy_raw = string_setting(settings, descriptor.proxy_attr)
        proxies = tuple(parse_api_keys(proxy_raw))

    base_url = string_setting(
        settings, descriptor.base_url_attr, descriptor.default_base_url or ""
    )
    resolved_base_url = base_url or descriptor.default_base_url
    if not resolved_base_url:
        raise AssertionError(
            f"Provider {descriptor.provider_id!r} has no configured base URL."
        )
    # Single proxy fallback (backward compat) — first entry from parsed list
    proxy = proxies[0] if proxies else ""
    return ProviderConfig(
        api_key=api_key,
        api_keys=api_keys,
        proxies=proxies,
        base_url=resolved_base_url,
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        enable_thinking=settings.enable_model_thinking,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )
