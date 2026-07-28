"""Tests for proxy pool storage, loading, and health filtering."""

import asyncio
from pathlib import Path

import pytest

from free_claude_code.config import proxy_pool
from free_claude_code.config.proxy_pool import (
    ProxyPoolEntry,
    _proxy_can_reach_target,
    load_healthy_proxy_urls,
    load_proxy_pool,
    save_proxy_pool,
)
from free_claude_code.config.proxy_pool import (
    test_all_pool_proxies as _test_all_pool_proxies,
)
from free_claude_code.config.proxy_pool import test_pool_proxy as _test_pool_proxy


def _patch_env(tmp_path: Path, monkeypatch, filename: str = ".env") -> Path:
    env_path = tmp_path / ".fcc" / filename
    env_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "free_claude_code.config.paths.managed_env_path",
        lambda: env_path,
    )
    return env_path


# --- Hermetic loopback fake-proxy harness for the real-tunnel probe ----------
#
# The old probe did a bare TCP connect to the proxy's own port, so a proxy that
# was listening but could not actually route outbound (corrupt SOCKS4 replies,
# filtered exit) still passed — and then got picked for grok login, where the
# CLI errored before printing a device URL (the 502). These tests stand up real
# loopback proxies that complete or refuse the tunnel handshake, and confirm
# the new probe distinguishes a working tunnel from a dead one.

# Canned IPv4 the SOCKS4/socks5-local probe resolves ``auth.x.ai`` to; patched
# so no real DNS/network is touched. The IP is arbitrary and unused on-wire.
_FAKE_IP = bytes([1, 2, 3, 4])


async def _fake_resolve(_host: str) -> bytes:
    return _FAKE_IP


async def _make_fake_proxy(behavior: str) -> tuple[asyncio.base_events.Server, int]:
    """Start a loopback proxy that emulates one probe outcome.

    ``behavior`` selects the canned protocol reply: ``"ok"`` (healthy tunnel:
    HTTP 200 / SOCKS5 success / SOCKS4 0x5A), ``"refused"`` (tunnel refused:
    HTTP 502 / SOCKS5 REP!=0 / SOCKS4 CD!=0x5A), ``"broken"`` (TCP-alive but
    corrupt bytes — the real-world dead-tunnel bug), or ``"drop"`` (accept
    then immediate EOF).
    """

    server = await asyncio.start_server(
        lambda r, w: _handle_proxy(r, w, behavior),
        "127.0.0.1",
        0,
    )
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _handle_proxy(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, behavior: str
) -> None:
    try:
        if behavior == "broken":
            # TCP-alive but sends corrupt bytes (no valid handshake reply). The
            # probe's readexactly(8)/(2) reads this garbage and returns False.
            writer.write(b"\x00\x01\x02 not a real socks/http reply")
            await writer.drain()
            return
        if behavior == "drop":
            return
        head = await reader.readexactly(2)
        if head[0] == 0x05:  # SOCKS5
            if behavior == "ok":
                writer.write(b"\x05\x00")
                await writer.drain()
                await reader.readexactly(4)
                writer.write(b"\x05\x00\x00\x01" + b"\x00" * 6)
                await writer.drain()
            else:
                writer.write(b"\x05\x05")
                await writer.drain()
        elif head[0] == 0x04:  # SOCKS4
            await reader.readexactly(6)
            if behavior == "ok":
                writer.write(b"\x00\x5a" + b"\x00" * 6)
                await writer.drain()
            else:
                writer.write(b"\x00\x5b" + b"\x00" * 6)
                await writer.drain()
        else:  # HTTP CONNECT
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            if behavior == "ok":
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()
    except asyncio.IncompleteReadError, OSError:
        pass
    finally:
        writer.close()


class TestProxyCanReachTarget:
    """The real-tunnel probe distinguishes working tunnels from dead ones."""

    @pytest.mark.asyncio
    async def test_empty_proxy_url_is_reachable(self, monkeypatch) -> None:
        # Empty URL == "go direct"; the probe treats it as trivially reachable.
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        assert await _proxy_can_reach_target("", timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_malformed_url_not_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        assert await _proxy_can_reach_target("not-a-url", timeout=1.0) is False

    @pytest.mark.asyncio
    async def test_unsupported_scheme_not_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        # ftp:// is not an http/socks scheme the probe speaks: it must report
        # unreachable rather than attempting a connect it can't handshake.
        assert await _proxy_can_reach_target("ftp://127.0.0.1:1", timeout=1.0) is False

    @pytest.mark.asyncio
    async def test_http_connect_ok_is_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("ok")
        try:
            url = f"http://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is True
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_http_connect_refused_not_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("refused")
        try:
            url = f"http://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is False
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_socks4_ok_is_reachable(self, monkeypatch) -> None:
        # SOCKS4 is the scheme httpx does NOT support — the probe handles it
        # with a raw handshake. This is the path the live dead proxies hit.
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("ok")
        try:
            url = f"socks4://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is True
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_socks4_corrupt_reply_not_reachable(self, monkeypatch) -> None:
        # The bug case: TCP-alive proxy that sends corrupt bytes instead of a
        # SOCKS4 reply. The old bare-TCP test passed this; the new probe fails it.
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("broken")
        try:
            url = f"socks4://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is False
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_socks4_refused_not_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("refused")
        try:
            url = f"socks4://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is False
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_socks5_ok_is_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("ok")
        try:
            url = f"socks5://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is True
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_socks5_refused_not_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("refused")
        try:
            url = f"socks5://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is False
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_tcp_alive_then_drop_not_reachable(self, monkeypatch) -> None:
        # Another dead-tunnel shape: accept then immediately EOF (no handshake
        # reply at all). Bare-TCP would call this reachable; the probe must not.
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("drop")
        try:
            url = f"socks5://127.0.0.1:{port}"
            assert await _proxy_can_reach_target(url, timeout=2.0) is False
        finally:
            server.close()
            await server.wait_closed()


class TestTestPoolProxy:
    """test_pool_proxy stamps the tunnel result onto the entry."""

    @pytest.mark.asyncio
    async def test_healthy_entry_set_false_for_dead_tunnel(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("broken")
        try:
            entry = ProxyPoolEntry(url=f"socks4://127.0.0.1:{port}", healthy=True)
            healthy = await _test_pool_proxy(entry, timeout=2.0)
            assert healthy is False
            assert entry.healthy is False
            assert entry.last_tested > 0.0
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_entry_set_true_for_working_tunnel(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        server, port = await _make_fake_proxy("ok")
        try:
            entry = ProxyPoolEntry(url=f"http://127.0.0.1:{port}", healthy=None)
            healthy = await _test_pool_proxy(entry, timeout=2.0)
            assert healthy is True
            assert entry.healthy is True
        finally:
            server.close()
            await server.wait_closed()


class TestTestAllPoolProxies:
    @pytest.mark.asyncio
    async def test_mixed_pool_marks_each_correctly(self, monkeypatch) -> None:
        monkeypatch.setattr(proxy_pool, "_resolve_ipv4", _fake_resolve)
        ok_server, ok_port = await _make_fake_proxy("ok")
        bad_server, bad_port = await _make_fake_proxy("broken")
        try:
            entries = [
                ProxyPoolEntry(url=f"http://127.0.0.1:{ok_port}"),
                ProxyPoolEntry(url=f"socks4://127.0.0.1:{bad_port}"),
            ]
            result = await _test_all_pool_proxies(entries, timeout=2.0)
            assert result[0].healthy is True
            assert result[1].healthy is False
        finally:
            ok_server.close()
            await ok_server.wait_closed()
            bad_server.close()
            await bad_server.wait_closed()


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
