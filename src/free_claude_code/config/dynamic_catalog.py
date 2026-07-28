"""Runtime catalog merging static PROVIDER_CATALOG with custom providers."""

from collections.abc import Iterator

from free_claude_code.config.custom_providers import (
    CustomProviderDefinition,
    load_custom_providers_from_managed_env,
)
from free_claude_code.config.grok_oauth import (
    GROK_OAUTH_DEFAULT_HEADERS,
    GROK_OAUTH_PROVIDER_ID,
)
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderDescriptor,
)


class DynamicProviderCatalog:
    """Merged catalog of static built-in providers and user-defined custom providers.

    Static entries (from ``PROVIDER_CATALOG``) take precedence: a custom
    provider whose id collides with a built-in provider is silently dropped.
    """

    def __init__(self) -> None:
        self._custom: dict[str, CustomProviderDefinition] = {}
        self._custom_descriptors: dict[str, ProviderDescriptor] = {}
        self._reload()

    def _reload(self) -> None:
        self._custom = load_custom_providers_from_managed_env()
        # Backfill the grok client headers for any grok_oauth definition that
        # predates them (so pre-existing accounts work after upgrade, and an
        # admin edit that rebuilds the definition without them is still served
        # the headers). In-memory only — does not churn the env file.
        grok = self._custom.get(GROK_OAUTH_PROVIDER_ID)
        if grok is not None and not grok.default_headers:
            self._custom[GROK_OAUTH_PROVIDER_ID] = CustomProviderDefinition(
                provider_id=grok.provider_id,
                display_name=grok.display_name,
                base_url=grok.base_url,
                api_keys=grok.api_keys,
                proxies=grok.proxies,
                detected_profile=grok.detected_profile,
                default_headers=GROK_OAUTH_DEFAULT_HEADERS,
            )
        self._custom_descriptors = {
            pid: d.to_descriptor() for pid, d in self._custom.items()
        }
        # Drop colliding ids
        for pid in list(self._custom_descriptors):
            if pid in PROVIDER_CATALOG:
                self._custom_descriptors.pop(pid, None)
                self._custom.pop(pid, None)

    def refresh(self) -> None:
        """Re-read custom providers from the managed env file."""
        self._reload()

    def resolve(self, provider_id: str) -> ProviderDescriptor | None:
        """Look up a provider descriptor in static or custom catalogs."""
        static = PROVIDER_CATALOG.get(provider_id)
        if static is not None:
            return static
        return self._custom_descriptors.get(provider_id)

    def is_dynamic(self, provider_id: str) -> bool:
        """Return whether *provider_id* is a user-added custom provider."""
        return (
            provider_id not in PROVIDER_CATALOG
            and provider_id in self._custom_descriptors
        )

    @property
    def custom_definitions(self) -> dict[str, CustomProviderDefinition]:
        """Return the raw custom provider definitions."""
        return dict(self._custom)

    @property
    def all_provider_ids(self) -> tuple[str, ...]:
        """Return all provider IDs: static first, then custom."""
        custom_ids = [
            pid for pid in self._custom_descriptors if pid not in PROVIDER_CATALOG
        ]
        return (*SUPPORTED_PROVIDER_IDS, *custom_ids)

    @property
    def custom_provider_ids(self) -> tuple[str, ...]:
        """Return only the custom provider IDs."""
        return tuple(self._custom_descriptors)

    def descriptor_iter(self) -> Iterator[ProviderDescriptor]:
        """Iterate over all descriptors: static then custom."""
        yield from PROVIDER_CATALOG.values()
        yield from self._custom_descriptors.values()

    def get_custom_definition(
        self, provider_id: str
    ) -> CustomProviderDefinition | None:
        """Return the full custom definition for *provider_id*, or None."""
        return self._custom.get(provider_id)
