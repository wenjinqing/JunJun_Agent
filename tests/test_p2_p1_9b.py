"""P2 小修 + P1-9b（MCP 持久 session）测试。

覆盖：
- MCP：holder 间接寻址（重连重绑不重注册）、连接失败降级
- scheduler：重入锁、cron 10 分钟迟到容忍、插件禁用跳过
- response_pool：Future 正常匹配 + 响应先于等待注册的暂存兜底
- link_preview：SSRF 私网/回环/云元数据拒绝
- jargon：写入后匹配缓存立即失效
- music：指定音源失败后降级其他源
"""

import asyncio
import time
from datetime import datetime

import pytest

# ---------------------------------------------------------------- MCP P1-9b


class _FakeTool:
    def __init__(self, name, coro):
        self.name = name
        self.coroutine = coro


class TestMcpHolderRebind:
    @pytest.mark.asyncio
    async def test_rebind_holder_redirects_wrapped_tool(self):
        """重连后 holder 指向新 coroutine，已注册的包装对象无需重注册。"""
        from junjun_mcp_client.client import MCPManager

        async def _old(**kw):
            return ([{"type": "text", "text": "旧 session"}], None)

        async def _new(**kw):
            return ([{"type": "text", "text": "新 session"}], None)

        mgr = MCPManager()
        tool = mgr._wrap(_FakeTool("some_tool", _old), "srv")

        content, _ = await tool.coroutine()
        assert content == "旧 session"

        # 模拟 _reconnect：按工具名重绑 holder
        holder = mgr._holders["srv"][0]
        assert holder["tool_name"] == "some_tool"
        holder["coro"] = _new

        content, _ = await tool.coroutine()
        assert content == "新 session"

    @pytest.mark.asyncio
    async def test_reconnect_rebinds_by_tool_name(self):
        """_reconnect 全流程：换新连接返回新工具后 holder 重绑，缺失工具保持旧绑定。"""
        from junjun_mcp_client.client import MCPManager

        mgr = MCPManager()
        mgr._configs["srv"] = {"command": "x"}
        mgr._holders["srv"] = [
            {"tool_name": "kept", "coro": "OLD_KEPT"},
            {"tool_name": "gone", "coro": "OLD_GONE"},
        ]

        async def fake_connect(name, cfg):
            return name, [_FakeTool("kept", "NEW_KEPT")]

        mgr._connect_one = fake_connect  # type: ignore[assignment]
        await mgr._reconnect("srv")

        holders = {h["tool_name"]: h["coro"] for h in mgr._holders["srv"]}
        assert holders["kept"] == "NEW_KEPT"
        assert holders["gone"] == "OLD_GONE"  # server 去掉了该工具：不动旧绑定

    @pytest.mark.asyncio
    async def test_connect_one_failure_degrades(self):
        """单 server 连接重试 3 次失败 -> 返回 (name, []) 不抛，不影响其他 server。"""
        from junjun_mcp_client.client import MCPManager

        mgr = MCPManager()
        name, tools = await mgr._connect_one("bad", {"command": "definitely_not_a_real_cmd_xx"})
        assert name == "bad" and tools == []

    @pytest.mark.asyncio
    async def test_holder_registered_per_server(self):
        """同一 manager 多个 server 的 holder 分桶存放。"""
        from junjun_mcp_client.client import MCPManager

        async def _ok(**kw):
            return "x", None

        mgr = MCPManager()
        mgr._wrap(_FakeTool("a", _ok), "srv1")
        mgr._wrap(_FakeTool("b", _ok), "srv2")
        assert [h["tool_name"] for h in mgr._holders["srv1"]] == ["a"]
        assert [h["tool_name"] for h in mgr._holders["srv2"]] == ["b"]


# ---------------------------------------------------------------- scheduler


class TestSchedulerP2:
    def test_reentry_lock_blocks_due(self):
        """上一次还没跑完（_running=True）不再判到期。"""
        from junjun_agent.loop.scheduler import ScheduledTask

        async def cb():
            pass

        t = ScheduledTask("t", cb, interval=1, _last_run=time.time() - 100)
        assert t.due()
        t._running = True
        assert not t.due()

    def test_cron_late_tolerance(self):
        """cron 错过分钟窗口 10 分钟内仍触发；超过 10 分钟当天放弃。"""
        from junjun_agent.loop.scheduler import _CRON_LATE_TOLERANCE, ScheduledTask

        async def cb():
            pass

        now = time.time()
        dt = datetime.fromtimestamp(now)
        scheduled = dt.replace(second=0, microsecond=0).timestamp()

        t = ScheduledTask("t", cb, cron_hour=dt.hour, cron_minute=dt.minute)
        assert t.due(now)  # 当分钟内

        # 迟到 5 分钟（容忍内）
        lag5 = scheduled + 300
        t2 = ScheduledTask("t2", cb, cron_hour=datetime.fromtimestamp(lag5).hour,
                           cron_minute=dt.minute)
        # cron 目标是 dt 的 HH:MM，5 分钟后同小时同分判定
        t2.cron_hour, t2.cron_minute = dt.hour, dt.minute
        assert t2.due(lag5)

        # 超过容忍窗口不触发
        beyond = scheduled + _CRON_LATE_TOLERANCE + 60
        t3 = ScheduledTask("t3", cb, cron_hour=dt.hour, cron_minute=dt.minute)
        assert not t3.due(beyond)

        # 当天已跑过不再触发
        t4 = ScheduledTask("t4", cb, cron_hour=dt.hour, cron_minute=dt.minute)
        t4.mark_run(now)
        assert not t4.due(now)

    @pytest.mark.asyncio
    async def test_plugin_disabled_skips_task(self):
        """插件被禁用时其后台任务不执行（禁用语义覆盖全生命周期）。"""
        from junjun_agent.loop.scheduler import ScheduledTask, Scheduler

        ran = []

        async def cb():
            ran.append(1)

        sched = Scheduler()
        task = ScheduledTask("pt", cb, plugin="some_plugin")
        from junjun_skills import registry
        registry._plugin_disabled.add("some_plugin")
        try:
            await sched._run_one(task)
            assert ran == []
        finally:
            registry._plugin_disabled.discard("some_plugin")

    @pytest.mark.asyncio
    async def test_run_one_releases_lock_on_exception(self):
        """任务抛异常：记 WARN 不上抛，重入锁释放（下轮还能跑）。"""
        from junjun_agent.loop.scheduler import ScheduledTask, Scheduler

        async def boom():
            raise RuntimeError("x")

        task = ScheduledTask("bt", boom)
        await Scheduler()._run_one(task)  # 不抛
        assert task._running is False


# ---------------------------------------------------------------- response_pool


class TestResponsePool:
    @pytest.mark.asyncio
    async def test_future_matched_by_echo(self):
        from junjun_adapter_napcat import response_pool as rp

        async def responder():
            await asyncio.sleep(0.01)
            await rp.put_response({"echo": "e1", "data": {"ok": 1}})

        task = asyncio.create_task(responder())
        result = await rp.get_response("e1", timeout=2)
        await task
        assert result["data"]["ok"] == 1

    @pytest.mark.asyncio
    async def test_early_response_stashed(self):
        """响应先于等待注册 -> 暂存，稍后 get_response 立即拿到。"""
        from junjun_adapter_napcat import response_pool as rp

        await rp.put_response({"echo": "early1", "data": 42})
        result = await rp.get_response("early1", timeout=2)
        assert result["data"] == 42
        assert "early1" not in rp._stash  # 取出后清除

    @pytest.mark.asyncio
    async def test_pending_cleaned_after_timeout(self):
        """超时后 _pending 不残留（防泄漏）。"""
        from junjun_adapter_napcat import response_pool as rp

        with pytest.raises(asyncio.TimeoutError):
            await rp.get_response("never", timeout=0.05)
        assert "never" not in rp._pending


# ---------------------------------------------------------------- link_preview SSRF


class TestLinkPreviewSsrf:
    def test_private_ip_rejected(self, monkeypatch):
        from junjun_memory import link_preview as lp

        monkeypatch.setattr("socket.getaddrinfo",
                            lambda host, port: [(2, 1, 6, "", ("192.168.1.1", 0))])
        assert lp._is_forbidden_target("http://internal.example/")

    def test_loopback_rejected(self, monkeypatch):
        from junjun_memory import link_preview as lp

        monkeypatch.setattr("socket.getaddrinfo",
                            lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))])
        assert lp._is_forbidden_target("http://localhost:8080/admin")

    def test_cloud_metadata_rejected(self, monkeypatch):
        """169.254.169.254 云元数据（link-local）拒绝。"""
        from junjun_memory import link_preview as lp

        monkeypatch.setattr("socket.getaddrinfo",
                            lambda host, port: [(2, 1, 6, "", ("169.254.169.254", 0))])
        assert lp._is_forbidden_target("http://169.254.169.254/latest/meta-data")

    def test_dns_failure_rejected(self, monkeypatch):
        from junjun_memory import link_preview as lp

        def _boom(host, port):
            raise OSError("name resolution failed")

        monkeypatch.setattr("socket.getaddrinfo", _boom)
        assert lp._is_forbidden_target("http://nonexistent.invalid/")

    def test_public_ip_allowed(self, monkeypatch):
        from junjun_memory import link_preview as lp

        monkeypatch.setattr("socket.getaddrinfo",
                            lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])
        assert not lp._is_forbidden_target("http://example.com/")


# ---------------------------------------------------------------- jargon 缓存失效


class TestJargonCacheInvalidation:
    @pytest.fixture(autouse=True)
    def _memory_db(self, monkeypatch):
        from peewee import SqliteDatabase

        import junjun_core.database.models as m
        test_db = SqliteDatabase(":memory:")
        tables = m.ALL_TABLES
        with test_db.bind_ctx(tables):
            test_db.create_tables(tables)
            monkeypatch.setattr(m, "db", test_db)
            import junjun_core.database as pkg
            monkeypatch.setattr(pkg, "db", test_db)
            yield test_db

    def test_record_invalidates_match_cache(self):
        from junjun_express import jargon as jg

        jg._invalidate_match_cache()
        jg.record_jargon("yyds", "永远的神")
        jg.record_jargon("yyds", "永远的神")  # count=2 才进匹配
        hits = jg.match_jargon_from_text("这波 yyds", "c1")
        assert hits and hits[0]["term"] == "yyds"

        # 缓存已热；写入新黑话后必须立即可见（不等 5 分钟 TTL）
        jg.record_jargon("绝绝子", "很棒")
        jg.record_jargon("绝绝子", "很棒")
        hits = jg.match_jargon_from_text("真是绝绝子", "c1")
        assert any(h["term"] == "绝绝子" for h in hits)
        jg._invalidate_match_cache()


# ---------------------------------------------------------------- music 降级


class TestMusicFallback:
    @pytest.mark.asyncio
    async def test_specified_source_falls_through(self, monkeypatch):
        """指定音源失败 -> 按降级顺序尝试其他源（原：指定失败直接没歌）。"""
        from junjun_skills.plugins.music import tools as mt

        async def fake_fetch(source, keyword, num=10):
            if source == "vip":
                return None
            if source == "qq":
                return [{"song": "晴天", "singer": "周杰伦"}]
            return None

        monkeypatch.setattr(mt, "fetch_search", fake_fetch)
        results, used = await mt._search_with_fallback("晴天", "vip")
        assert used == "qq" and results[0]["song"] == "晴天"

    @pytest.mark.asyncio
    async def test_no_source_tries_all_in_order(self, monkeypatch):
        from junjun_skills.plugins.music import tools as mt

        calls = []

        async def fake_fetch(source, keyword, num=10):
            calls.append(source)
            return [{"song": "s"}] if source == "juhe" else None

        monkeypatch.setattr(mt, "fetch_search", fake_fetch)
        results, used = await mt._search_with_fallback("x")
        assert used == "juhe"
        assert calls == ["netease", "qq", "vip", "juhe"]

    @pytest.mark.asyncio
    async def test_all_fail_returns_none(self, monkeypatch):
        from junjun_skills.plugins.music import tools as mt

        async def fake_fetch(source, keyword, num=10):
            return None

        monkeypatch.setattr(mt, "fetch_search", fake_fetch)
        results, used = await mt._search_with_fallback("x")
        assert results is None and used is None
