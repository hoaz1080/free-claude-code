"""Proxy pool management — central list of proxies shared across all providers.

Proxies are stored in the managed env file under ``FCC_PROXY_POOL`` as a JSON
array. Each entry has a ``url``, optional ``label``, and ``healthy`` status
(set by health checks). Healthy proxies are automatically used by all providers
as a shared rotation pool. When the pool is empty, providers fall back to their
individual proxy env vars (backward compatible).
"""

import asyncio
import contextlib
import json
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

FCC_PROXY_POOL_KEY = "FCC_PROXY_POOL"

# Real-tunnel probe target. The grok device-code login talks to
# ``auth.x.ai`` (https://auth.x.ai/oauth2/device/code) and the user approves at
# ``accounts.x.ai``; a proxy is only useful to grok login if it can actually
# open a tunnel to that host. A bare TCP connect to the *proxy's own* port
# passes for proxies that are listening but cannot route outbound (corrupt
# SOCKS4 replies, filtered exits, …) — those then get picked for login and the
# grok CLI errors before it can print a device URL (the 502). Proving the tunnel
# end-to-end is what makes the health check accurate.
_TUNNEL_TEST_HOST = "auth.x.ai"
_TUNNEL_TEST_PORT = 443


def _default_proxy_port(scheme: str) -> int:
    return 1080 if scheme in ("socks5", "socks5h", "socks4", "socks4a") else 80


async def _resolve_ipv4(host: str) -> bytes | None:
    """Local-resolve *host* to a 4-byte IPv4, or ``None`` on failure."""
    try:
        infos = socket.getaddrinfo(host, _TUNNEL_TEST_PORT, socket.AF_INET)
    except OSError:
        return None
    for info in infos:
        # AF_INET family → first address element is the IPv4 string, but the
        # getaddrinfo stub returns a family-union sockaddr, so reach for it
        # defensively rather than declaring a narrower tuple type.
        ip = info[4][0]
        if not isinstance(ip, str):
            continue
        octets = ip.split(".")
        if len(octets) == 4:
            try:
                return bytes(int(o) for o in octets)
            except ValueError:
                continue
    return None


async def _http_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    timeout: float,
) -> bool:
    """HTTP CONNECT to the tunnel target — success iff proxy answers ``2xx``."""
    request = (
        f"CONNECT {_TUNNEL_TEST_HOST}:{_TUNNEL_TEST_PORT} HTTP/1.1\r\n"
        f"Host: {_TUNNEL_TEST_HOST}:{_TUNNEL_TEST_PORT}\r\n\r\n"
    ).encode()
    try:
        writer.write(request)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    except TimeoutError, OSError:
        return False
    parts = line.split()
    return len(parts) >= 2 and parts[1].startswith(b"2")


async def _socks5_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    scheme: str,
    timeout: float,
) -> bool:
    """SOCKS5 (+h) CONNECT — ``socks5`` resolves locally, ``socks5h`` remotely."""
    try:
        writer.write(b"\x05\x01\x00")  # VER, 1 method, no-auth
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        reply = await asyncio.wait_for(reader.readexactly(2), timeout=timeout)
    except TimeoutError, OSError, asyncio.IncompleteReadError:
        return False
    if reply != b"\x05\x00":  # must select no-auth
        return False
    port_be = _TUNNEL_TEST_PORT.to_bytes(2, "big")
    if scheme == "socks5h":
        host_b = _TUNNEL_TEST_HOST.encode()
        request = b"\x05\x01\x00\x03" + bytes([len(host_b)]) + host_b + port_be
    else:
        ip = await _resolve_ipv4(_TUNNEL_TEST_HOST)
        if ip is None:
            return False
        request = b"\x05\x01\x00\x01" + ip + port_be
    try:
        writer.write(request)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        head = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
    except TimeoutError, OSError, asyncio.IncompleteReadError:
        return False
    return head[0] == 0x05 and head[1] == 0x00  # VER, REP(success)


async def _socks4_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    timeout: float,
) -> bool:
    """SOCKS4 (IP-only) CONNECT — reqwest/grok resolve locally, so we mirror that."""
    ip = await _resolve_ipv4(_TUNNEL_TEST_HOST)
    if ip is None:
        return False
    request = b"\x04\x01" + _TUNNEL_TEST_PORT.to_bytes(2, "big") + ip + b"\x00"
    try:
        writer.write(request)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        reply = await asyncio.wait_for(reader.readexactly(8), timeout=timeout)
    except TimeoutError, OSError, asyncio.IncompleteReadError:
        return False
    # Reply: VN(0) + CD; CD == 0x5A == request granted.
    return reply[0] == 0x00 and reply[1] == 0x5A


async def _proxy_can_reach_target(proxy_url: str, timeout: float = 8.0) -> bool:
    """True iff *proxy_url* can establish an outbound tunnel to the target.

    Replaces the old bare-TCP-connect-to-the-proxy test: a proxy whose listen
    port is open but which cannot actually route to ``auth.x.ai:443`` (corrupt
    SOCKS4 replies, filtered exit, dead tunnel) now reports ``False``. Speaks
    HTTP CONNECT, SOCKS5(+h) and SOCKS4 over a single asyncio stream — no extra
    dependency, and uniform success semantics across schemes. ``socks4://`` is
    handled directly because httpx has no SOCKS4 support.
    """
    if not proxy_url:
        return True
    parsed = urlsplit(proxy_url)
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or _default_proxy_port(scheme)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except TimeoutError, OSError:
        return False
    try:
        if scheme in ("http", "https"):
            return await _http_connect(reader, writer, timeout)
        if scheme in ("socks5", "socks5h"):
            return await _socks5_connect(reader, writer, scheme, timeout)
        if scheme in ("socks4", "socks4a"):
            return await _socks4_connect(reader, writer, timeout)
        return False
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


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


async def test_pool_proxy(entry: ProxyPoolEntry, timeout: float = 8.0) -> bool:
    """Test one proxy and return True if it can really tunnel to the target."""
    healthy = await _proxy_can_reach_target(entry.url, timeout=timeout)
    entry.healthy = healthy
    entry.last_tested = time.time()
    return healthy


async def test_all_pool_proxies(
    entries: list[ProxyPoolEntry], timeout: float = 8.0
) -> list[ProxyPoolEntry]:
    """Test all proxies in the pool concurrently (real tunnel probes)."""

    async def test_one(entry: ProxyPoolEntry) -> None:
        entry.healthy = await _proxy_can_reach_target(entry.url, timeout=timeout)
        entry.last_tested = time.time()

    tasks = [test_one(e) for e in entries if e.url]
    if tasks:
        await asyncio.gather(*tasks)
    return entries
