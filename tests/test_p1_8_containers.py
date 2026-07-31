"""P1-8 无界容器治理测试：会话淘汰 / 队列条目回收 / db_writer 有界+看门狗 /
昵称缓存上限 / Agent 连接池关闭。
"""

import asyncio
import time
from types import SimpleNamespace

import pytest


class TestSessionEviction:
    def _manager_with_sessions(self, specs):
        """specs: [(chat_id, last_active_ts)]"""
        from junjun_core.gateway.session_manager import ChatSessionManager
        mgr = ChatSessionManager()
        from junjun_core.gateway.session_manager import ChatSession
        for chat_id, ts in specs:
            s = ChatSession(chat_id, "qq", group_id="1")
            s.last_active_ts = ts
            mgr._sessions[chat_id] = s
        return mgr

    def test_idle_sessions_evicted(self):
        now = time.time()
        mgr = self._manager_with_sessions([
            ("qq:1:group", now - 4 * 86400),   # 4 天没消息 -> 淘汰
            ("qq:2:group", now - 3600),        # 1 小时前 -> 保留
        ])
        evicted = mgr.evict_idle(ttl=3 * 86400)
        assert evicted == 1
        assert "qq:1:group" not in mgr.all_sessions()
        assert "qq:2:group" in mgr.all_sessions()

    def test_fresh_session_never_evicted(self):
        """刚创建（last_active_ts=0）的会话不被 TTL 淘汰。"""
        mgr = self._manager_with_sessions([("qq:new:group", 0.0)])
        assert mgr.evict_idle(ttl=1.0) == 0
        assert "qq:new:group" in mgr.all_sessions()

    def test_max_sessions_evicts_oldest(self):
        now = time.time()
        mgr = self._manager_with_sessions([
            (f"qq:{i}:group", now - i * 100) for i in range(5)
        ])
        evicted = mgr.evict_idle(ttl=999 * 86400, max_sessions=2)
        assert evicted == 3
        remaining = mgr.all_sessions()
        assert len(remaining) == 2
        assert "qq:0:group" in remaining  # 最活跃的留下

    def test_on_evict_callback_invoked(self):
        now = time.time()
        mgr = self._manager_with_sessions([("qq:1:group", now - 9 * 86400)])
        called = []
        mgr.on_evict = lambda s: called.append(s.chat_id)
        mgr.evict_idle(ttl=3 * 86400)
        assert called == ["qq:1:group"]

    @pytest.mark.asyncio
    async def test_agent_pool_closed_on_evict(self):
        """被淘汰会话的 agent.aclose() 被调度执行（连接池不泄漏）。"""
        now = time.time()
        mgr = self._manager_with_sessions([("qq:1:group", now - 9 * 86400)])
        closed = []

        class _Agent:
            async def aclose(self):
                closed.append(True)
        mgr._sessions["qq:1:group"].agent = _Agent()
        mgr.evict_idle(ttl=3 * 86400)
        await asyncio.sleep(0.05)  # 让 create_task 跑完
        assert closed


class TestSessionQueueRecycle:
    @pytest.mark.asyncio
    async def test_done_worker_entry_recycled(self):
        """worker 已退出且队列空的旧条目被回收重建（有未消费消息的条目保留重启）。"""
        from junjun_agent.funnel.session_queue import SessionQueueManager, SessionQueue
        mgr = SessionQueueManager()
        session = SimpleNamespace(chat_id="qq:1:group")

        async def _noop_handler(*a):
            return None

        q1 = SessionQueue("qq:1:group", _noop_handler)
        q1.start()  # 真实 worker 起再杀，得到 done()=True 的任务
        q1._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await q1._task
        mgr._queues["qq:1:group"] = q1

        mgr.dispatch(session, SimpleNamespace(text="again"), _noop_handler)
        assert mgr._queues["qq:1:group"] is not q1  # 空队列死 worker -> 重建
        await mgr.stop_all()

    @pytest.mark.asyncio
    async def test_dead_worker_with_pending_kept_and_restarted(self):
        """队列里还有消息的旧条目不回收（不丢消息），put 时 worker 自动重启。"""
        from junjun_agent.funnel.session_queue import SessionQueueManager, SessionQueue
        mgr = SessionQueueManager()
        session = SimpleNamespace(chat_id="qq:1:group")

        async def _noop_handler(*a):
            return None

        q1 = SessionQueue("qq:1:group", _noop_handler)
        q1.start()
        q1._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await q1._task
        q1._queue.put_nowait((session, SimpleNamespace(text="pending"), time.time()))
        mgr._queues["qq:1:group"] = q1

        mgr.dispatch(session, SimpleNamespace(text="again"), _noop_handler)
        assert mgr._queues["qq:1:group"] is q1  # 有消息 -> 保留
        assert q1._task is not None and not q1._task.done()  # worker 已重启
        await mgr.stop_all()

    def test_drop_only_when_idle(self):
        from junjun_agent.funnel.session_queue import SessionQueueManager, SessionQueue
        mgr = SessionQueueManager()
        q = SessionQueue("qq:1:group", lambda *a: None)
        mgr._queues["qq:1:group"] = q
        mgr.drop("qq:1:group")  # 无任务无消息 -> 清
        assert "qq:1:group" not in mgr._queues


class TestDBWriterHardening:
    @pytest.mark.asyncio
    async def test_queue_full_drops_with_warn(self):
        from junjun_core.database.writer import DBWriter
        w = DBWriter()
        w._queue = asyncio.Queue(maxsize=2)
        w._task = SimpleNamespace(done=lambda: False)  # 伪装活着
        for i in range(5):
            w.submit(lambda: None)
        assert w._queue.qsize() == 2  # 满了丢最新，不炸

    @pytest.mark.asyncio
    async def test_dead_worker_restarts(self):
        """写协程死亡后 submit 自动重启，后续写入仍被消费。"""
        from junjun_core.database.writer import DBWriter
        w = DBWriter()
        w.start()
        # 杀死写协程（模拟异常逃逸）
        w._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await w._task
        done = []
        w.submit(lambda: done.append(1))
        assert w._task is not None and not w._task.done()
        await asyncio.sleep(0.1)
        assert done == [1]  # 重启后正常消费
        await w.stop()

    @pytest.mark.asyncio
    async def test_executor_timeout_unblocks_loop(self):
        """卡死的写入不永久挂住协程：超时跳过后继续消费下一条。"""
        import junjun_core.database.writer as writer_mod
        old = writer_mod._EXEC_TIMEOUT
        writer_mod._EXEC_TIMEOUT = 0.1
        try:
            w = writer_mod.DBWriter()
            w.start()
            done = []

            def _stuck():
                time.sleep(5)

            w.submit(_stuck)
            w.submit(lambda: done.append(1))
            for _ in range(30):
                await asyncio.sleep(0.1)
                if done:
                    break
            assert done == [1]
            await w.stop()
        finally:
            writer_mod._EXEC_TIMEOUT = old


class TestNickCacheBound:
    def test_cache_evicts_expired_then_oldest(self):
        import importlib
        mh = importlib.import_module(
            "junjun_adapter_napcat.recv_handler.message_handler")
        mh._NICK_CACHE.clear()
        now = time.time()
        # 填满：一半过期一半新鲜
        for i in range(mh._NICK_CACHE_MAX // 2):
            mh._NICK_CACHE[("g", f"old{i}")] = (f"n{i}", now - 7200)
            mh._NICK_CACHE[("g", f"new{i}")] = (f"n{i}", now)
        mh._nick_cache_put(("g", "incoming"), "name")
        assert len(mh._NICK_CACHE) <= mh._NICK_CACHE_MAX
        assert ("g", "incoming") in mh._NICK_CACHE
        # 过期项应已被清掉
        assert ("g", "old0") not in mh._NICK_CACHE
        mh._NICK_CACHE.clear()


class TestAgentClose:
    @pytest.mark.asyncio
    async def test_aclose_closes_model_clients(self):
        from junjun_agent.agent import JunJunAgent
        closed = []

        class _AsyncClient:
            async def close(self):
                closed.append(1)

        class _Model:
            async_client = _AsyncClient()

        agent = JunJunAgent.__new__(JunJunAgent)
        agent._model = _Model()
        await agent.aclose()
        assert closed == [1]

    @pytest.mark.asyncio
    async def test_aclose_tolerates_weird_models(self):
        from junjun_agent.agent import JunJunAgent
        agent = JunJunAgent.__new__(JunJunAgent)
        agent._model = object()  # 无 async_client
        await agent.aclose()  # 不炸
