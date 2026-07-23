"""Tests for ApiKeyPool multi-key rotation."""

import pytest

from free_claude_code.providers.key_pool import (
    ApiKeyEntry,
    ApiKeyPool,
    mask_key,
    mask_proxy,
)
from free_claude_code.providers.key_pool import (
    test_proxy_connectivity as _test_proxy_connectivity,
)


class TestApiKeyPoolInit:
    def test_single_key(self) -> None:
        pool = ApiKeyPool(["sk-test"])
        assert pool.current_key == "sk-test"
        assert pool.has_available_key is True

    def test_multiple_keys(self) -> None:
        pool = ApiKeyPool(["key1", "key2", "key3"])
        assert pool.current_key == "key1"
        assert pool.has_available_key is True

    def test_strips_whitespace(self) -> None:
        pool = ApiKeyPool(["  key1  ", " key2 "])
        assert pool.current_key == "key1"

    def test_empty_key_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one non-empty key"):
            ApiKeyPool([])

    def test_all_whitespace_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one non-empty key"):
            ApiKeyPool(["  ", "  "])


class TestApiKeyPoolRotation:
    def test_rotate_moves_to_next_healthy_key(self) -> None:
        pool = ApiKeyPool(["key1", "key2", "key3"])
        # _rotate scans forward from current index, so key1<e2><b6><ab>key2<e2><86><92>key3→key1
        pool._rotate()
        assert pool.current_key == "key2"
        pool._rotate()
        assert pool.current_key == "key3"
        pool._rotate()
        assert pool.current_key == "key1"

    def test_mark_rate_limited_sets_cooldown_and_rotates(self) -> None:
        pool = ApiKeyPool(["key1", "key2", "key3"])
        next_key = pool.mark_rate_limited(30.0)
        # key1 cooldown set, rotates to key2
        assert next_key == "key2"
        assert pool.current_key == "key2"

    def test_rate_limited_key_blocked_during_cooldown(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        pool.mark_rate_limited(3600.0)  # key1 on long cooldown
        # Rotate past key2 back — key1 blocked so picks key2 (shortest cooldown)
        pool._rotate()
        # Both are blocked: key1 cooldown=3600, key2 cooldown=0 (healthy)
        # So current_key returns key2 (the healthy one)
        assert pool.current_key == "key2"

    def test_mark_auth_failed_permanently_disables_and_rotates(self) -> None:
        pool = ApiKeyPool(["key1", "key2", "key3"])
        next_key = pool.mark_auth_failed()
        # key1 permanently dead, rotates to key2
        assert next_key == "key2"
        assert pool.current_key == "key2"
        # Rotate twice: key2→key3→key1(skip,dead)→key2
        pool._rotate()
        assert pool.current_key == "key3"
        pool._rotate()
        assert pool.current_key == "key2"


class TestApiKeyPoolExhaustion:
    def test_all_permanently_failed_returns_none(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        pool.mark_auth_failed()
        pool.mark_auth_failed()
        assert pool.current_key is None
        assert pool.has_available_key is False

    def test_all_rate_limited_returns_shortest_cooldown(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        pool.mark_rate_limited(3600.0)  # long cooldown on key1
        pool.mark_rate_limited(5.0)  # short cooldown on key2
        # Both on cooldown — current_key scans for available, finds none,
        # then returns key with shortest remaining cooldown (key2)
        assert pool.current_key == "key2"

    def test_mixed_failed_and_rate_limited_returns_rate_limited(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        pool.mark_auth_failed()  # key1 dead → rotates to key2
        pool.mark_rate_limited(3600.0)  # key2 on cooldown → tries to rotate
        # key1 is dead, key2 on cooldown — key2 is only non-dead option
        assert pool.current_key == "key2"
        assert pool.has_available_key is True


class TestApiKeyPoolReset:
    def test_reset_cooldowns_clears_temporary_blocks(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        pool.mark_rate_limited(3600.0)  # key1 blocked → rotates to key2
        assert pool.current_key == "key2"
        pool.reset_cooldowns()  # key1 cooldown cleared
        pool._rotate()
        # key1 healthy again after reset
        assert pool.current_key == "key1"

    def test_reset_cooldowns_does_not_revive_permanently_failed(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        pool.mark_auth_failed()  # key1 dead → rotates to key2
        pool.mark_rate_limited(3600.0)  # key2 on cooldown → can't rotate
        pool.reset_cooldowns()  # key2 cooldown cleared (but key1 still dead)
        assert pool.current_key == "key2"  # key2 now healthy
        pool._rotate()
        # wraps to key1 (dead) → stays at key2
        assert pool.current_key == "key2"


class TestApiKeyPoolWithProxies:
    def test_pool_with_proxies(self) -> None:
        pool = ApiKeyPool(
            ["key1", "key2", "key3"],
            proxies=["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"],
        )
        assert pool.current_key == "key1"
        assert pool.current_proxy == "http://proxy1:8080"
        assert pool.available_count == 3

    def test_proxy_fewer_than_keys_pads_with_empty(self) -> None:
        pool = ApiKeyPool(
            ["key1", "key2", "key3"],
            proxies=["http://proxy1:8080"],
        )
        assert pool.current_key == "key1"
        assert pool.current_proxy == "http://proxy1:8080"
        pool._rotate()
        assert pool.current_key == "key2"
        assert pool.current_proxy == ""  # padded

    def test_proxy_more_than_keys_truncated(self) -> None:
        pool = ApiKeyPool(
            ["key1", "key2"],
            proxies=["http://proxy1:8080", "http://proxy2:8080", "http://extra:3128"],
        )
        assert pool.current_key == "key1"
        assert pool.current_proxy == "http://proxy1:8080"
        pool._rotate()
        assert pool.current_key == "key2"
        assert pool.current_proxy == "http://proxy2:8080"

    def test_no_proxies_returns_empty_string(self) -> None:
        pool = ApiKeyPool(["key1", "key2"])
        assert pool.current_proxy == ""

    def test_rotate_prefers_different_proxy_on_rate_limit(self) -> None:
        pool = ApiKeyPool(
            ["key1", "key2", "key3"],
            proxies=["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"],
        )
        # key1 with proxy1 is rate limited → should rotate to key2 (different proxy)
        pool.mark_rate_limited(30.0)
        # After rotate: key1 blocked, now on key2
        assert pool.current_key == "key2"
        assert pool.current_proxy == "http://proxy2:8080"

    def test_proxy_unhealthy_marks_entry_unavailable(self) -> None:
        entry = ApiKeyEntry(key="test-key", proxy="http://bad-proxy:9999")
        assert entry.available is True
        entry.proxy_unhealthy = True
        assert entry.available is False

    def test_pool_skips_proxy_unhealthy_entries(self) -> None:
        pool = ApiKeyPool(
            ["key1", "key2", "key3"],
            proxies=["http://bad:3128", "http://proxy2:8080", "http://proxy3:8080"],
            health_check_proxies=False,
        )
        # Force proxy1 unhealthy
        pool._entries[0].proxy_unhealthy = True
        assert pool.current_key == "key2"  # auto-skips to key2

    def test_current_key_id_masks_key(self) -> None:
        pool = ApiKeyPool(["sk-1234567890abcdef"], health_check_proxies=False)
        eid = pool.current_key_id
        assert "sk-123" in eid
        assert "abcdef" not in eid

    def test_no_available_key_returns_empty_proxy(self) -> None:
        pool = ApiKeyPool(["key1"], health_check_proxies=False)
        pool.mark_auth_failed()
        assert pool.current_key is None
        assert pool.current_proxy == ""
        assert pool.current_key_id == "unknown"


class TestMaskHelpers:
    def test_mask_key_short(self) -> None:
        assert mask_key("abc") == "abc****"

    def test_mask_key_long(self) -> None:
        masked = mask_key("sk-1234567890abcdef")
        assert masked.startswith("sk-123")
        assert "..." in masked
        assert "cdef" in masked

    def test_mask_proxy_no_password(self) -> None:
        assert mask_proxy("http://proxy:8080") == "http://proxy:8080"

    def test_mask_proxy_with_password(self) -> None:
        masked = mask_proxy("http://user:pass@proxy:8080")
        assert "pass" not in masked
        assert "****" in masked

    def test_mask_proxy_empty(self) -> None:
        assert mask_proxy("") == ""


class TestProxyConnectivity:
    @pytest.mark.asyncio
    async def test_empty_proxy_is_healthy(self) -> None:
        assert await _test_proxy_connectivity("") is True

    @pytest.mark.asyncio
    async def test_unreachable_proxy_returns_false(self) -> None:
        result = await _test_proxy_connectivity("http://127.0.0.1:1", timeout=1.0)
        assert result is False
