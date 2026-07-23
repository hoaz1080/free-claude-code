"""Custom provider definitions, key parsing, and managed-env persistence."""

import contextlib
import json
from dataclasses import dataclass

from loguru import logger

from free_claude_code.config.detection import (
    detect_provider_profile,
    generate_provider_id,
)
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.provider_catalog import ProviderDescriptor

CUSTOM_PROVIDERS_ENV_KEY = "FCC_CUSTOM_PROVIDERS"


@dataclass(frozen=True, slots=True)
class CustomProviderDefinition:
    """A user-defined provider with base_url, keys, proxies, and detected profile.

    Each key can have an associated proxy URL. When keys are rotated on
    rate-limit (429), the pool prefers a key with a *different* proxy so
    the retry goes through a different IP.
    """

    provider_id: str
    display_name: str
    base_url: str
    api_keys: tuple[str, ...]
    proxies: tuple[str, ...] = ()
    detected_profile: str | None = (
        None  # profile id from detection, or None for generic
    )

    def to_descriptor(self) -> ProviderDescriptor:
        """Convert to a catalog-compatible descriptor."""
        return ProviderDescriptor(
            provider_id=self.provider_id,
            display_name=self.display_name,
            default_base_url=self.base_url,
            credential_env=None,  # custom providers don't use env vars for keys
            credential_attr=None,
            credential_url=None,
            proxy_attr=None,
            static_credential=None,  # keys come from the CustomProviderDefinition
        )


def parse_api_keys(raw: str) -> list[str]:
    """Split comma-separated API keys, stripping whitespace and empties.

    ``""`` returns ``[]``, ``"key1"`` returns ``["key1"]``,
    ``"key1, key2, key3 "`` returns ``["key1", "key2", "key3"]``.
    """
    if not raw or not raw.strip():
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def load_custom_providers_from_managed_env() -> dict[str, CustomProviderDefinition]:
    """Read custom provider definitions from the managed env file.

    The managed env file contains a ``FCC_CUSTOM_PROVIDERS`` key whose
    value is a JSON array of custom provider objects. Returns a mapping
    of provider_id → CustomProviderDefinition.
    """
    path = managed_env_path()
    if not path.is_file():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    json_value = _extract_dotenv_key(content, CUSTOM_PROVIDERS_ENV_KEY)
    if json_value is None:
        return {}

    return _parse_custom_providers_json(json_value)


def _extract_dotenv_key(content: str, key: str) -> str | None:
    """Extract the value of *key* from dotenv-format *content*.

    Handles quoted and unquoted values. Returns None if key not found.
    """
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, v = stripped.partition("=")
        if k.strip() != key:
            continue
        value = v.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return value
    return None


def _parse_custom_providers_json(
    json_value: str,
) -> dict[str, CustomProviderDefinition]:
    """Parse custom providers from a JSON string.

    Expected format: a JSON array of objects with:
    ``display_name`` (optional), ``base_url`` (required),
    ``api_keys`` (array of strings, required).
    """
    try:
        raw_definitions: list[dict] = json.loads(json_value)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Invalid FCC_CUSTOM_PROVIDERS JSON: {}", exc)
        return {}

    if not isinstance(raw_definitions, list):
        logger.warning(
            "FCC_CUSTOM_PROVIDERS must be a JSON array, got {}",
            type(raw_definitions).__name__,
        )
        return {}

    result: dict[str, CustomProviderDefinition] = {}
    for item in raw_definitions:
        if not isinstance(item, dict):
            continue
        base_url = item.get("base_url", "").strip()
        if not base_url:
            continue
        api_keys_raw = item.get("api_keys", [])
        if not isinstance(api_keys_raw, list):
            continue
        api_keys = tuple(k.strip() for k in api_keys_raw if k.strip())
        if not api_keys:
            continue

        proxies_raw = item.get("proxies")
        if isinstance(proxies_raw, list):
            proxies = tuple(p.strip() for p in proxies_raw if p.strip())
        else:
            proxies = ()

        provider_id = item.get("provider_id", "").strip() or generate_provider_id(
            base_url
        )
        display_name = (
            item.get("display_name", "").strip()
            or provider_id.replace("_", " ").title()
        )
        detected_profile = item.get("detected_profile") or detect_provider_profile(
            base_url
        )

        result[provider_id] = CustomProviderDefinition(
            provider_id=provider_id,
            display_name=display_name,
            base_url=base_url,
            api_keys=api_keys,
            proxies=proxies,
            detected_profile=detected_profile,
        )

    return result


def save_custom_providers_to_managed_env(
    definitions: dict[str, CustomProviderDefinition],
) -> None:
    """Persist custom provider definitions to the managed env file.

    Merges with existing managed env content: replaces the
    ``FCC_CUSTOM_PROVIDERS`` line while preserving all other keys.
    """
    path = managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = ""
    if path.is_file():
        with contextlib.suppress(OSError):
            existing_content = path.read_text(encoding="utf-8")

    json_array = [
        {
            "provider_id": d.provider_id,
            "display_name": d.display_name,
            "base_url": d.base_url,
            "api_keys": list(d.api_keys),
            "proxies": list(d.proxies),
            "detected_profile": d.detected_profile,
        }
        for d in definitions.values()
    ]
    json_value = json.dumps(json_array, ensure_ascii=False, separators=(",", ":"))

    new_line = f"{CUSTOM_PROVIDERS_ENV_KEY}={_quote_dotenv_value(json_value)}"

    # Replace existing FCC_CUSTOM_PROVIDERS line or append
    lines = existing_content.split("\n") if existing_content else []
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(CUSTOM_PROVIDERS_ENV_KEY + "=") or stripped.startswith(
            f"#{CUSTOM_PROVIDERS_ENV_KEY}"
        ):
            lines[i] = new_line
            replaced = True
            break

    if not replaced:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("# Custom providers (managed by /admin)")
        lines.append(new_line)

    new_content = "\n".join(lines) + "\n"

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(new_content, encoding="utf-8")
        import os

        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)

    logger.info(
        "Saved {} custom provider(s) to {}",
        len(definitions),
        path,
    )


def _quote_dotenv_value(value: str) -> str:
    """Quote a dotenv value that contains special characters."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def merge_legacy_keys(
    descriptor: ProviderDescriptor,
    raw_credential: str,
) -> tuple[str, ...]:
    """Resolve the effective API key tuple for a static provider.

    When *raw_credential* contains commas, it is split into multiple keys.
    Single-key values and static credentials (local providers) return a
    one-element tuple.
    """
    if not raw_credential or not raw_credential.strip():
        return ()
    keys = parse_api_keys(raw_credential)
    if not keys:
        return ()
    return tuple(keys)
