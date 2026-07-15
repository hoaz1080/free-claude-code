"""Tests for ApiKeyPool multi-key rotation."""

import pytest

from free_claude_code.providers.key_pool import ApiKeyPool


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
        # _rotate scans forward from current index, so key1→key2→key3→key1
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
