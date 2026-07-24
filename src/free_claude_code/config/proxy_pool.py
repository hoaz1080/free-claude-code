"""Proxy pool management — central list of proxies shared across all providers.

Proxies are stored in the managed env file under ``FCC_PROXY_POOL`` as a JSON
array. Each entry has a ``url``, optional ``label``, and ``healthy`` status
(set by health checks). Healthy proxies are automatically used by all providers
as a shared rotation pool. When the pool is empty, providers fall back to their
individual proxy env vars (backward compatible).
"""

import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


async def _tcp_connect(host: str, port: int, timeout: float = 5.0) -> bool:
    """TCP connect test helper — local copy to avoid cross-package imports."""
    import asyncio

    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


FCC_PROXY_POOL_KEY = "FCC_PROXY_POOL"


@dataclass
class ProxyPoolEntry:
    """One proxy in the shared pool."""

    url: str
    label: str = ""
    healthy: bool | None = None  # None = untested
    last_tested: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "label": self.label,
            "healthy": self.healthy,
            "last_tested": self.last_tested,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProxyPoolEntry:
        return cls(
            url=data["url"],
            label=data.get("label", ""),
            healthy=data.get("healthy"),
            last_tested=data.get("last_tested", 0.0),
        )


def _managed_env_path():
    """Lazy import to avoid circular imports at module level."""
    from free_claude_code.config.paths import managed_env_path

    return managed_env_path()


def _read_managed_env() -> str:
    path = _managed_env_path()
    if path.is_file():
        with contextlib.suppress(OSError):
            return path.read_text(encoding="utf-8")
    return ""


def _write_managed_env(content: str) -> None:
    path = _managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        import os

        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_dotenv_key(content: str, key: str) -> str | None:
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


def _quote_dotenv_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def load_proxy_pool() -> list[ProxyPoolEntry]:
    """Load proxy pool from managed env."""
    content = _read_managed_env()
    if not content:
        return []

    json_value = _extract_dotenv_key(content, FCC_PROXY_POOL_KEY)
    if json_value is None:
        return []

    try:
        raw: list[dict] = json.loads(json_value)
        if not isinstance(raw, list):
            return []
        return [
            ProxyPoolEntry.from_dict(item)
            for item in raw
            if isinstance(item, dict) and item.get("url")
        ]
    except json.JSONDecodeError, TypeError:
        logger.warning("Invalid FCC_PROXY_POOL JSON")
        return []


def save_proxy_pool(entries: list[ProxyPoolEntry]) -> None:
    """Save proxy pool to managed env."""
    json_array = [e.to_dict() for e in entries]
    json_value = json.dumps(json_array, ensure_ascii=False, separators=(",", ":"))
    new_line = f"{FCC_PROXY_POOL_KEY}={_quote_dotenv_value(json_value)}"

    content = _read_managed_env()
    lines = content.split("\n") if content else []
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(FCC_PROXY_POOL_KEY + "=") or stripped.startswith(
            f"#{FCC_PROXY_POOL_KEY}"
        ):
            lines[i] = new_line
            replaced = True
            break

    if not replaced:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("# Proxy pool (managed by /admin)")
        lines.append(new_line)

    _write_managed_env("\n".join(lines) + "\n")
    logger.info("Saved {} proxy pool entries", len(entries))


def load_healthy_proxy_urls() -> tuple[str, ...]:
    """Return URLs of proxies marked healthy (or untested) in the pool."""
    entries = load_proxy_pool()
    # Include untested ones too (optimistic — they'll fail fast if bad)
    healthy = [e.url for e in entries if e.healthy is not False]
    return tuple(healthy)


async def test_pool_proxy(entry: ProxyPoolEntry, timeout: float = 5.0) -> bool:
    """Test one proxy and return True if reachable."""
    healthy = await _test_url_reachable(entry.url, timeout=timeout)
    entry.healthy = healthy
    entry.last_tested = time.time()
    return healthy


async def _test_url_reachable(proxy_url: str, timeout: float = 5.0) -> bool:
    """Test if a proxy URL is reachable via TCP connect."""
    from urllib.parse import urlsplit

    if not proxy_url:
        return True
    try:
        parsed = urlsplit(proxy_url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port
        if port is None:
            scheme = parsed.scheme
            port = (
                1080
                if scheme in ("socks5", "socks5h")
                else 80
                if scheme == "http"
                else 443
            )
        return await _tcp_connect(host, port, timeout=timeout)
    except Exception:
        return False


async def test_all_pool_proxies(
    entries: list[ProxyPoolEntry], timeout: float = 5.0
) -> list[ProxyPoolEntry]:
    """Test all proxies in the pool concurrently."""
    import asyncio

    async def test_one(entry: ProxyPoolEntry) -> None:
        entry.healthy = await _test_url_reachable(entry.url, timeout=timeout)
        entry.last_tested = time.time()

    tasks = [test_one(e) for e in entries if e.url]
    if tasks:
        await asyncio.gather(*tasks)
    return entries
