"""Tests for URL-based provider detection."""

from free_claude_code.config.detection import (
    HOST_PROFILE_MAP,
    detect_provider_profile,
    generate_provider_id,
)


class TestDetectProviderProfile:
    def test_known_host_deepseek(self) -> None:
        assert detect_provider_profile("https://api.deepseek.com/v1") == "deepseek"

    def test_known_host_groq(self) -> None:
        assert detect_provider_profile("https://api.groq.com/openai/v1") == "groq"

    def test_known_host_mistral(self) -> None:
        assert detect_provider_profile("https://api.mistral.ai/v1") == "mistral"

    def test_known_host_openrouter(self) -> None:
        assert detect_provider_profile("https://openrouter.ai/api/v1") == "open_router"

    def test_unknown_host_returns_none(self) -> None:
        assert detect_provider_profile("https://my-custom-llm.example.com/v1") is None

    def test_localhost_with_port_returns_none(self) -> None:
        assert detect_provider_profile("http://localhost:8000/v1") is None

    def test_openai_not_in_map_returns_none(self) -> None:
        # api.openai.com is deliberately not in the profile map — generic fallback
        assert detect_provider_profile("https://api.openai.com/v1") is None

    def test_empty_url_returns_none(self) -> None:
        assert detect_provider_profile("") is None

    def test_url_without_host_returns_none(self) -> None:
        assert detect_provider_profile("/v1/chat/completions") is None

    def test_trailing_slash_normalized(self) -> None:
        assert detect_provider_profile("https://api.deepseek.com/") == "deepseek"


class TestGenerateProviderId:
    def test_generates_from_hostname(self) -> None:
        pid = generate_provider_id("https://api.openai.com/v1")
        assert pid == "custom_api_openai_com"

    def test_dots_and_dashes_become_underscores(self) -> None:
        pid = generate_provider_id("https://my-llm.example.com/v1")
        assert pid == "custom_my_llm_example_com"

    def test_subdomain_preserved(self) -> None:
        pid = generate_provider_id("https://api.deepseek.com/v1")
        assert pid == "custom_api_deepseek_com"

    def test_port_included(self) -> None:
        pid = generate_provider_id("http://localhost:8080/v1")
        assert pid == "custom_localhost_8080"

    def test_empty_url_returns_generic(self) -> None:
        pid = generate_provider_id("")
        assert pid.startswith("custom_")


class TestHostProfileMapIntegrity:
    def test_all_values_are_strings(self) -> None:
        for host, profile in HOST_PROFILE_MAP.items():
            assert isinstance(host, str)
            assert isinstance(profile, str)
            assert profile, f"empty profile for host {host!r}"

    def test_no_duplicate_keys(self) -> None:
        assert len(set(HOST_PROFILE_MAP)) == len(HOST_PROFILE_MAP)
