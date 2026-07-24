"""Tests for proxy pool storage, loading, and health filtering."""

from pathlib import Path

from free_claude_code.config.proxy_pool import (
    ProxyPoolEntry,
    load_healthy_proxy_urls,
    load_proxy_pool,
    save_proxy_pool,
)


def _patch_env(tmp_path: Path, monkeypatch, filename: str = ".env") -> Path:
    env_path = tmp_path / ".fcc" / filename
    env_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "free_claude_code.config.paths.managed_env_path",
        lambda: env_path,
    )
    return env_path


class TestProxyPoolEntry:
    def test_to_dict_roundtrips_through_from_dict(self) -> None:
        entry = ProxyPoolEntry(
            url="socks5://174.77.111.198:49547",
            label="home",
            healthy=True,
            last_tested=12345.0,
        )
        restored = ProxyPoolEntry.from_dict(entry.to_dict())
        assert restored == entry

    def test_from_dict_uses_defaults_for_missing_optional_fields(self) -> None:
        entry = ProxyPoolEntry.from_dict({"url": "http://proxy:8080"})
        assert entry.url == "http://proxy:8080"
        assert entry.label == ""
        assert entry.healthy is None
        assert entry.last_tested == 0.0


class TestSaveAndLoad:
    def test_roundtrip_preserves_urls(self, tmp_path: Path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        entries = [
            ProxyPoolEntry(url="http://proxy1:8080", label="work"),
            ProxyPoolEntry(url="socks5://174.77.111.198:49547", label="home"),
            ProxyPoolEntry(url="https://user:p@ss@host:443"),
        ]
        save_proxy_pool(entries)
        loaded = load_proxy_pool()
        assert [e.url for e in loaded] == [e.url for e in entries]
        assert [e.label for e in loaded] == ["work", "home", ""]

    def test_load_from_empty_file_returns_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text("")
        assert load_proxy_pool() == []

    def test_load_from_missing_file_returns_empty(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _patch_env(tmp_path, monkeypatch, filename="missing/.env")
        assert load_proxy_pool() == []

    def test_invalid_json_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text('FCC_PROXY_POOL="not valid json"\n')
        assert load_proxy_pool() == []

    def test_non_list_json_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text('FCC_PROXY_POOL="{}"\n')
        assert load_proxy_pool() == []

    def test_save_replaces_existing_pool_line(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text("EXISTING_KEY=value\nFCC_PROXY_POOL=old\n")
        save_proxy_pool([ProxyPoolEntry(url="http://new:8080")])
        content = env_path.read_text()
        assert "old" not in content.split("FCC_PROXY_POOL")[1]
        assert "http://new:8080" in content
        # Other keys preserved
        assert "EXISTING_KEY=value" in content

    def test_save_appends_when_no_existing_line(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        env_path = _patch_env(tmp_path, monkeypatch)
        env_path.write_text("OTHER=value\n")
        save_proxy_pool([ProxyPoolEntry(url="http://new:8080")])
        content = env_path.read_text()
        assert "FCC_PROXY_POOL=" in content
        assert "OTHER=value" in content

    def test_save_skips_entries_without_url(self, tmp_path: Path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        # Directly stash malformed entries (an empty url entry should be
        # filtered out by load, not crash save)
        save_proxy_pool([ProxyPoolEntry(url="http://ok:8080")])
        assert len(load_proxy_pool()) == 1


class TestLoadHealthyProxyUrls:
    def test_includes_healthy_and_untested(self, tmp_path: Path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        save_proxy_pool(
            [
                ProxyPoolEntry(url="http://healthy:8080", healthy=True),
                ProxyPoolEntry(url="http://untested:8080", healthy=None),
                ProxyPoolEntry(url="http://dead:8080", healthy=False),
            ]
        )
        urls = load_healthy_proxy_urls()
        assert "http://healthy:8080" in urls
        assert "http://untested:8080" in urls
        assert "http://dead:8080" not in urls

    def test_empty_pool_returns_empty_tuple(self, tmp_path: Path, monkeypatch) -> None:
        _patch_env(tmp_path, monkeypatch)
        assert load_healthy_proxy_urls() == ()
