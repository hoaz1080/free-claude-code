"""Tests for custom provider parsing, loading, and persistence."""

from pathlib import Path

from free_claude_code.config.custom_providers import (
    CustomProviderDefinition,
    load_custom_providers_from_managed_env,
    merge_legacy_keys,
    parse_api_keys,
    save_custom_providers_to_managed_env,
)
from free_claude_code.config.provider_catalog import ProviderDescriptor


class TestParseApiKeys:
    def test_single_key(self) -> None:
        assert parse_api_keys("sk-test") == ["sk-test"]

    def test_comma_separated(self) -> None:
        assert parse_api_keys("key1, key2, key3") == ["key1", "key2", "key3"]

    def test_trims_whitespace(self) -> None:
        assert parse_api_keys("  a , b , c  ") == ["a", "b", "c"]

    def test_filters_empty_entries(self) -> None:
        assert parse_api_keys("key1,,key2, ,key3") == ["key1", "key2", "key3"]

    def test_empty_string_returns_empty(self) -> None:
        assert parse_api_keys("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert parse_api_keys("   ") == []


class TestMergeLegacyKeys:
    def test_single_key_unchanged(self) -> None:
        descriptor = ProviderDescriptor(
            provider_id="test",
            display_name="Test",
            default_base_url="https://test.example.com/v1",
            credential_env="TEST_KEY",
        )
        result = merge_legacy_keys(descriptor, "sk-abc123")
        assert result == ("sk-abc123",)

    def test_comma_separated_splits(self) -> None:
        descriptor = ProviderDescriptor(
            provider_id="test",
            display_name="Test",
            default_base_url="https://test.example.com/v1",
            credential_env="TEST_KEY",
        )
        result = merge_legacy_keys(descriptor, "sk-a, sk-b, sk-c")
        assert result == ("sk-a", "sk-b", "sk-c")

    def test_empty_credential_returns_empty(self) -> None:
        descriptor = ProviderDescriptor(
            provider_id="test",
            display_name="Test",
            default_base_url="https://test.example.com/v1",
            credential_env="TEST_KEY",
        )
        assert merge_legacy_keys(descriptor, "") == ()
        assert merge_legacy_keys(descriptor, "  ") == ()


class TestCustomProviderDefinition:
    def test_to_descriptor(self) -> None:
        definition = CustomProviderDefinition(
            provider_id="custom_openai_com",
            display_name="Custom OpenAI",
            base_url="https://api.openai.com/v1",
            api_keys=("sk-a", "sk-b"),
            detected_profile=None,
        )
        descriptor = definition.to_descriptor()
        assert descriptor.provider_id == "custom_openai_com"
        assert descriptor.display_name == "Custom OpenAI"
        assert descriptor.default_base_url == "https://api.openai.com/v1"


class TestCustomProviderPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch) -> None:
        """Save custom providers to a temp managed env file, then load them back."""
        env_path = tmp_path / ".fcc" / ".env"
        monkeypatch.setattr(
            "free_claude_code.config.custom_providers.managed_env_path",
            lambda: env_path,
        )
        # Clear any existing global state by saving empty first
        save_custom_providers_to_managed_env({})

        definitions = {
            "custom_test": CustomProviderDefinition(
                provider_id="custom_test",
                display_name="Test Provider",
                base_url="https://test.example.com/v1",
                api_keys=("sk-test1", "sk-test2"),
                detected_profile=None,
            )
        }
        save_custom_providers_to_managed_env(definitions)

        loaded = load_custom_providers_from_managed_env()
        assert len(loaded) == 1
        assert "custom_test" in loaded
        loaded_def = loaded["custom_test"]
        assert loaded_def.provider_id == "custom_test"
        assert loaded_def.display_name == "Test Provider"
        assert loaded_def.base_url == "https://test.example.com/v1"
        assert loaded_def.api_keys == ("sk-test1", "sk-test2")

    def test_load_from_empty_file_returns_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = tmp_path / ".fcc" / ".env"
        env_path.parent.mkdir(parents=True)
        env_path.write_text("SOME_OTHER_KEY=value\n")
        monkeypatch.setattr(
            "free_claude_code.config.custom_providers.managed_env_path",
            lambda: env_path,
        )
        result = load_custom_providers_from_managed_env()
        assert result == {}

    def test_load_from_missing_file_returns_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = tmp_path / ".fcc" / "nonexistent" / ".env"
        monkeypatch.setattr(
            "free_claude_code.config.custom_providers.managed_env_path",
            lambda: env_path,
        )
        result = load_custom_providers_from_managed_env()
        assert result == {}

    def test_save_merges_with_existing_env_content(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = tmp_path / ".fcc" / ".env"
        env_path.parent.mkdir(parents=True)
        env_path.write_text("EXISTING_KEY=existing_value\n")
        monkeypatch.setattr(
            "free_claude_code.config.custom_providers.managed_env_path",
            lambda: env_path,
        )

        definitions = {
            "custom_test": CustomProviderDefinition(
                provider_id="custom_test",
                display_name="Test",
                base_url="https://test.example.com/v1",
                api_keys=("sk-test",),
                detected_profile=None,
            )
        }
        save_custom_providers_to_managed_env(definitions)

        content = env_path.read_text()
        assert "EXISTING_KEY=existing_value" in content
        assert "FCC_CUSTOM_PROVIDERS=" in content
        assert "custom_test" in content

    def test_save_deletes_and_readds_provider(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = tmp_path / ".fcc" / ".env"
        monkeypatch.setattr(
            "free_claude_code.config.custom_providers.managed_env_path",
            lambda: env_path,
        )

        defs_before = {
            "custom_a": CustomProviderDefinition(
                provider_id="custom_a",
                display_name="A",
                base_url="https://a.example.com/v1",
                api_keys=("sk-a",),
                detected_profile=None,
            ),
            "custom_b": CustomProviderDefinition(
                provider_id="custom_b",
                display_name="B",
                base_url="https://b.example.com/v1",
                api_keys=("sk-b",),
                detected_profile=None,
            ),
        }
        save_custom_providers_to_managed_env(defs_before)
        assert len(load_custom_providers_from_managed_env()) == 2

        # Remove custom_b
        del defs_before["custom_b"]
        save_custom_providers_to_managed_env(defs_before)
        loaded = load_custom_providers_from_managed_env()
        assert len(loaded) == 1
        assert "custom_a" in loaded
        assert "custom_b" not in loaded
