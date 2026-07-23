"""Multi-key pool with automatic rotation on rate-limit and auth failures.

Each key can have an associated proxy URL. When the pool rotates on failure,
it picks the next key+proxy pair so the retry goes through a different IP.
Proxy health is checked lazily when first used.
"""

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from loguru import logger

DEFAULT_RATE_LIMIT_COOLDOWN = 60.0
PROXY_CONNECT_TIMEOUT = 5.0


def mask_key(key: str) -> str:
    """Return a masked version of an API key for logging."""
    if len(key) <= 8:
        return key[:4] + "****"
    return key[:6] + "..." + key[-4:]


def mask_proxy(proxy_url: str) -> str:
    """Mask password in a proxy URL for logging."""
    if not proxy_url:
        return ""
    try:
        parsed = urlsplit(proxy_url)
        if parsed.password:
            return proxy_url.replace(parsed.password, "****")
    except Exception:
        pass
    return proxy_url


async def test_proxy_connectivity(
    proxy_url: str, timeout: float = PROXY_CONNECT_TIMEOUT
) -> bool:
    """Test if a proxy is reachable via TCP connect.

    Returns True if the proxy is reachable or *proxy_url* is empty.
    """
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
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


@dataclass(slots=True)
class ApiKeyEntry:
    """One API key, its proxy, and health state."""

    key: str
    proxy: str = ""
    cooldown_until: float = 0.0
    permanently_failed: bool = False
    proxy_unhealthy: bool = False

    @property
    def available(self) -> bool:
        if self.permanently_failed:
            return False
        if self.proxy_unhealthy:
            return False
        return time.monotonic() >= self.cooldown_until

    @property
    def remaining_cooldown(self) -> float:
        if self.permanently_failed:
            return float("inf")
        return max(0.0, self.cooldown_until - time.monotonic())

    @property
    def key_id(self) -> str:
        """Short masked identifier for error messages."""
        return mask_key(self.key)

    @property
    def proxy_id(self) -> str:
        """Masked proxy URL for logging."""
        return mask_proxy(self.proxy)


class ApiKeyPool:
    """Ordered pool of API keys with automatic rotation on failure.

    Each key may have an associated proxy URL. When a key hits rate limit
    (429), it gets a cooldown and the pool rotates to the next available
    key+proxy pair. When a key has an auth failure (401/403), it is
    permanently disabled. On rotate, a key with a *different* proxy is
    preferred so the retry goes through a different IP.
    """

    def __init__(
        self,
        keys: list[str],
        proxies: list[str] | None = None,
        *,
        health_check_proxies: bool = True,
    ) -> None:
        cleaned_keys = [k.strip() for k in keys if k.strip()]
        if not cleaned_keys:
            raise ValueError("ApiKeyPool requires at least one non-empty key")
        cleaned_proxies = [p.strip() for p in proxies if p.strip()] if proxies else []
        self._entries = [
            ApiKeyEntry(
                key=k, proxy=cleaned_proxies[i] if i < len(cleaned_proxies) else ""
            )
            for i, k in enumerate(cleaned_keys)
        ]
        self._index = 0
        self._has_ran_health_check = False
        self._health_check_proxies = health_check_proxies
        logger.info(
            "ApiKeyPool initialized: keys={}, proxies={}",
            len(self._entries),
            sum(1 for e in self._entries if e.proxy),
        )

    @property
    def current_key(self) -> str | None:
        """Return the first available key, or the one with shortest cooldown."""
        entry = self._current_entry()
        return entry.key if entry is not None else None

    @property
    def current_proxy(self) -> str:
        """Return the proxy of the current entry."""
        entry = self._current_entry()
        return entry.proxy if entry is not None else ""

    @property
    def current_key_id(self) -> str:
        """Return masked identifier of the current key for error messages."""
        entry = self._current_entry()
        return entry.key_id if entry is not None else "unknown"

    @property
    def has_available_key(self) -> bool:
        """Return whether at least one key is not permanently failed."""
        return any(
            not e.permanently_failed and not e.proxy_unhealthy for e in self._entries
        )

    @property
    def available_count(self) -> int:
        """Return the number of non-permanently-failed keys."""
        return sum(
            1
            for e in self._entries
            if not e.permanently_failed and not e.proxy_unhealthy
        )

    @property
    def _all_permanently_failed(self) -> bool:
        return all(e.permanently_failed for e in self._entries)

    async def run_proxy_health_checks(self) -> None:
        """Test all proxies and mark unhealthy ones.

        Proxies that fail the connectivity test are marked as
        ``proxy_unhealthy`` so they are skipped during rotation.
        """
        if not self._health_check_proxies:
            self._has_ran_health_check = True
            return
        if self._has_ran_health_check:
            return
        self._has_ran_health_check = True

        tasks = []
        for i, entry in enumerate(self._entries):
            if entry.proxy:
                tasks.append(self._check_one_proxy(i, entry))
        if tasks:
            await asyncio.gather(*tasks)

    async def _check_one_proxy(self, index: int, entry: ApiKeyEntry) -> None:
        healthy = await test_proxy_connectivity(entry.proxy)
        if not healthy:
            entry.proxy_unhealthy = True
            logger.warning(
                "ApiKeyPool: proxy unhealthy for key {} (proxy={})",
                entry.key_id,
                entry.proxy_id,
            )
        else:
            logger.debug(
                "ApiKeyPool: proxy OK for key {} (proxy={})",
                entry.key_id,
                entry.proxy_id,
            )

    def mark_rate_limited(
        self, cooldown_seconds: float = DEFAULT_RATE_LIMIT_COOLDOWN
    ) -> str | None:
        """Mark the current key as rate-limited and rotate.

        Returns the next available key, or None if all keys are exhausted.
        """
        if self._all_permanently_failed:
            return None
        entry = self._entry_at(self._index)
        if entry is None:
            return None
        entry.cooldown_until = time.monotonic() + max(0.0, cooldown_seconds)
        logger.warning(
            "ApiKeyPool: key {} rate-limited for {:.1f}s, rotating",
            entry.key_id,
            cooldown_seconds,
        )
        self._rotate(prefer_different_proxy=bool(entry.proxy))
        return self.current_key

    def mark_auth_failed(self) -> str | None:
        """Permanently disable the current key and rotate.

        Returns the next available key, or None if all keys are exhausted.
        """
        entry = self._entry_at(self._index)
        if entry is None:
            return None
        entry.permanently_failed = True
        logger.error(
            "ApiKeyPool: key {} permanently failed (auth), rotating",
            entry.key_id,
        )
        self._rotate(prefer_different_proxy=bool(entry.proxy))
        return self.current_key

    def reset_cooldowns(self) -> None:
        """Clear all temporary cooldowns (e.g. after a successful request)."""
        for entry in self._entries:
            if not entry.permanently_failed:
                entry.cooldown_until = 0.0

    def _rotate(self, *, prefer_different_proxy: bool = False) -> None:
        """Move index to the next available key.

        When *prefer_different_proxy* is True, among available keys, prefer
        one whose proxy differs from the current entry's proxy. This ensures
        that rate-limited requests retry through a different IP.
        """
        current_proxy = self._entries[self._index].proxy if self._entries else ""
        scanned = 0
        n = len(self._entries)

        while scanned < n:
            self._index = (self._index + 1) % n
            scanned += 1
            entry = self._entries[self._index]
            if entry.available:
                # If we want a different proxy but this one has the same proxy,
                # keep scanning (unless we've gone full circle)
                if (
                    prefer_different_proxy
                    and entry.proxy
                    and entry.proxy == current_proxy
                ):
                    continue
                return

        # All keys are blocked or failed; pick the one with shortest cooldown
        best_index = min(
            range(n),
            key=lambda i: self._entries[i].remaining_cooldown,
        )
        self._index = best_index

    def _current_entry(self) -> ApiKeyEntry | None:
        """Return the first available entry, or best-effort fallback."""
        entry = self._entries[self._index]
        if entry.available:
            return entry
        # Scan for any available key
        for i, e in enumerate(self._entries):
            if e.available:
                self._index = i
                return e
        # All blocked/failed — return shortest cooldown
        if self._all_permanently_failed:
            return None
        best_index = min(
            range(len(self._entries)),
            key=lambda i: self._entries[i].remaining_cooldown,
        )
        self._index = best_index
        return self._entries[best_index]

    def _entry_at(self, index: int) -> ApiKeyEntry | None:
        if index < 0 or index >= len(self._entries):
            return None
        return self._entries[index]

    def __repr__(self) -> str:
        available = sum(1 for e in self._entries if e.available)
        failed = sum(1 for e in self._entries if e.permanently_failed)
        proxy_unhealthy = sum(1 for e in self._entries if e.proxy_unhealthy)
        blocked = len(self._entries) - available - failed - proxy_unhealthy
        proxies = sum(1 for e in self._entries if e.proxy)
        return (
            f"ApiKeyPool(total={len(self._entries)}, "
            f"available={available}, blocked={blocked}, "
            f"failed={failed}, proxy_unhealthy={proxy_unhealthy}, "
            f"proxies={proxies})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ApiKeyPool):
            return NotImplemented
        return self._entries == other._entries and self._index == other._index

    def __hash__(self) -> int:
        return hash((tuple(self._entries), self._index))
