"""Tests for DynamicProviderCatalog merging static + custom providers."""

from unittest.mock import patch

from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.dynamic_catalog import DynamicProviderCatalog
from free_claude_code.config.provider_catalog import (
    SUPPORTED_PROVIDER_IDS,
)


class TestDynamicCatalogEmpty:
    def test_resolve_static_provider(self) -> None:
        catalog = DynamicProviderCatalog()
        descriptor = catalog.resolve("deepseek")
        assert descriptor is not None
        assert descriptor.provider_id == "deepseek"

    def test_resolve_unknown_returns_none(self) -> None:
        catalog = DynamicProviderCatalog()
        assert catalog.resolve("nonexistent") is None

    def test_all_provider_ids_includes_static(self) -> None:
        catalog = DynamicProviderCatalog()
        assert set(catalog.all_provider_ids) == set(SUPPORTED_PROVIDER_IDS)

    def test_custom_provider_ids_empty_initially(self) -> None:
        catalog = DynamicProviderCatalog()
        assert catalog.custom_provider_ids == ()

    def test_is_dynamic_false_for_static(self) -> None:
        catalog = DynamicProviderCatalog()
        assert catalog.is_dynamic("deepseek") is False


class TestDynamicCatalogWithCustom:
    @staticmethod
    def _custom_def(
        provider_id: str = "custom_test",
        base_url: str = "https://test.example.com/v1",
        api_keys: tuple[str, ...] = ("sk-test",),
        detected_profile: str | None = None,
    ) -> CustomProviderDefinition:
        return CustomProviderDefinition(
            provider_id=provider_id,
            display_name="Test Provider",
            base_url=base_url,
            api_keys=api_keys,
            detected_profile=detected_profile,
        )

    def _patch_custom_providers(self, definitions):
        """Patch at the DynamicProviderCatalog level where refresh() imports."""
        return patch.object(
            DynamicProviderCatalog,
            "_DynamicProviderCatalog__load_custom",
            return_value=definitions,
            create=True,
        )

    def test_refresh_adds_custom_providers(self) -> None:
        defs = {"custom_test": self._custom_def()}
        catalog = DynamicProviderCatalog()
        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value=defs,
        ):
            catalog.refresh()
        assert catalog.is_dynamic("custom_test") is True
        assert catalog.resolve("custom_test") is not None

    def test_custom_provider_ids_after_refresh(self) -> None:
        defs = {
            "custom_a": self._custom_def(provider_id="custom_a"),
            "custom_b": self._custom_def(provider_id="custom_b"),
        }
        catalog = DynamicProviderCatalog()
        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value=defs,
        ):
            catalog.refresh()
        assert set(catalog.custom_provider_ids) == {"custom_a", "custom_b"}

    def test_all_provider_ids_includes_custom(self) -> None:
        defs = {"custom_x": self._custom_def(provider_id="custom_x")}
        catalog = DynamicProviderCatalog()
        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value=defs,
        ):
            catalog.refresh()
        all_ids = set(catalog.all_provider_ids)
        assert "custom_x" in all_ids
        assert set(SUPPORTED_PROVIDER_IDS).issubset(all_ids)

    def test_static_takes_precedence_over_custom_collision(self) -> None:
        defs = {"deepseek": self._custom_def(provider_id="deepseek")}
        catalog = DynamicProviderCatalog()
        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value=defs,
        ):
            catalog.refresh()
        assert catalog.is_dynamic("deepseek") is False
        descriptor = catalog.resolve("deepseek")
        assert descriptor is not None
        assert descriptor.credential_env is not None

    def test_is_dynamic_true_for_custom(self) -> None:
        defs = {"custom_z": self._custom_def(provider_id="custom_z")}
        catalog = DynamicProviderCatalog()
        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value=defs,
        ):
            catalog.refresh()
        assert catalog.is_dynamic("custom_z") is True

    def test_fresh_catalog_is_empty_before_refresh(self) -> None:
        catalog = DynamicProviderCatalog()
        assert catalog.custom_provider_ids == ()

    def test_multiple_refreshes_replace_custom_providers(self) -> None:
        catalog = DynamicProviderCatalog()
        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value={"custom_1": self._custom_def(provider_id="custom_1")},
        ):
            catalog.refresh()
        assert catalog.custom_provider_ids == ("custom_1",)

        with patch(
            "free_claude_code.config.dynamic_catalog.load_custom_providers_from_managed_env",
            return_value={"custom_2": self._custom_def(provider_id="custom_2")},
        ):
            catalog.refresh()
        assert catalog.custom_provider_ids == ("custom_2",)
        assert catalog.resolve("custom_1") is None
