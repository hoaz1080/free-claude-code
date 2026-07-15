"""Multi-key pool with automatic rotation on rate-limit and auth failures."""

import time
from dataclasses import dataclass

from loguru import logger

DEFAULT_RATE_LIMIT_COOLDOWN = 60.0


@dataclass(slots=True)
class ApiKeyEntry:
    """One API key and its health state."""

    key: str
    cooldown_until: float = 0.0
    permanently_failed: bool = False

    @property
    def available(self) -> bool:
        if self.permanently_failed:
            return False
        return time.monotonic() >= self.cooldown_until

    @property
    def remaining_cooldown(self) -> float:
        if self.permanently_failed:
            return float("inf")
        return max(0.0, self.cooldown_until - time.monotonic())


class ApiKeyPool:
    """Ordered pool of API keys with automatic rotation on failure.

    When a key hits rate limit (429), it gets a cooldown and the pool
    rotates to the next available key. When a key has an auth failure
    (401/403), it is permanently disabled. Requests pick the first
    available key in order.
    """

    def __init__(self, keys: list[str]) -> None:
        cleaned = [k.strip() for k in keys if k.strip()]
        if not cleaned:
            raise ValueError("ApiKeyPool requires at least one non-empty key")
        self._entries = [ApiKeyEntry(key=k) for k in cleaned]
        self._index = 0
        logger.info(
            "ApiKeyPool initialized: keys={}",
            len(self._entries),
        )

    @property
    def current_key(self) -> str | None:
        """Return the first available key, or the one with shortest cooldown."""
        entry = self._current_entry()
        return entry.key if entry is not None else None

    @property
    def has_available_key(self) -> bool:
        """Return whether at least one key is not permanently failed."""
        return any(not e.permanently_failed for e in self._entries)

    @property
    def available_count(self) -> int:
        """Return the number of non-permanently-failed keys."""
        return sum(1 for e in self._entries if not e.permanently_failed)

    @property
    def _all_permanently_failed(self) -> bool:
        return all(e.permanently_failed for e in self._entries)

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
            "ApiKeyPool: key rate-limited for {:.1f}s, rotating",
            cooldown_seconds,
        )
        self._rotate()
        return self.current_key

    def mark_auth_failed(self) -> str | None:
        """Permanently disable the current key and rotate.

        Returns the next available key, or None if all keys are exhausted.
        """
        entry = self._entry_at(self._index)
        if entry is None:
            return None
        entry.permanently_failed = True
        logger.error("ApiKeyPool: key permanently failed (auth), rotating")
        self._rotate()
        return self.current_key

    def reset_cooldowns(self) -> None:
        """Clear all temporary cooldowns (e.g. after a successful request)."""
        for entry in self._entries:
            if not entry.permanently_failed:
                entry.cooldown_until = 0.0

    def _rotate(self) -> None:
        """Move index to the next available key."""
        scanned = 0
        n = len(self._entries)
        while scanned < n:
            self._index = (self._index + 1) % n
            scanned += 1
            if self._entries[self._index].available:
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
        blocked = len(self._entries) - available - failed
        return (
            f"ApiKeyPool(total={len(self._entries)}, "
            f"available={available}, blocked={blocked}, failed={failed})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ApiKeyPool):
            return NotImplemented
        return self._entries == other._entries and self._index == other._index

    def __hash__(self) -> int:
        return hash((tuple(self._entries), self._index))
