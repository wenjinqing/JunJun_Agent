"""异步任务管理器测试：提交/去重/超时/异常兜底/完成直发/优雅退出。"""

import asyncio

import pytest

from junjun_agent.tasks import TaskManager, _parse_route
from junjun_core.contracts import ReplySegment


@pytest.fixture
def sent(monkeypatch):
    """拦截 gateway 直发。"""
    box = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            box.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return box


@pytest.fixture
def mgr():
    return TaskManager()


CHAT = "qq:12345:group"


class TestRouteParse:
    def test_group(self):
        assert _parse_route("qq:12345:group") == ("qq", None, "12345")

    def test_private(self):
        assert _parse_route("qq:678:private") == ("qq", "678", None)

    def test_malformed(self):
        platform, uid, gid = _parse_route("")
        assert gid is None and uid is None


class TestSubmit:
    @pytest.mark.asyncio
    async def test_success_sends_segments(self, mgr, sent):
        async def work():
            return [ReplySegment(type="image", data="http://x/1.png")]

        ack = await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        assert "在弄了" in ack
        await asyncio.gather(*mgr._running.values())
        assert len(sent) == 1
        segs = sent[0].segments
        assert segs[-1].type == "image"
        assert sent[0].target_group_id == "12345"

    @pytest.mark.asyncio
    async def test_done_text_prepended(self, mgr, sent):
        async def work():
            return [ReplySegment(type="image", data="http://x/1.png")]

        await mgr.submit(kind="ai_draw", work=work, done_text="画好了！", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        segs = sent[0].segments
        assert segs[0].type == "text" and segs[0].data == "画好了！"

    @pytest.mark.asyncio
    async def test_failure_sends_fail_text(self, mgr, sent):
        async def work():
            return None

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画失败了"

    @pytest.mark.asyncio
    async def test_exception_caught_sends_fail_text(self, mgr, sent):
        async def work():
            raise RuntimeError("boom")

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画失败了"

    @pytest.mark.asyncio
    async def test_dedup_busy(self, mgr, sent):
        gate = asyncio.Event()

        async def work():
            await gate.wait()
            return [ReplySegment(type="text", data="done")]

        ack1 = await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        assert "在弄了" in ack1
        ack2 = await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        assert ack2 != ack1  # 占线话术
        assert len(mgr._running) == 1
        gate.set()
        await asyncio.gather(*mgr._running.values(), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_different_kind_not_blocked(self, mgr, sent):
        gate = asyncio.Event()

        async def slow():
            await gate.wait()
            return [ReplySegment(type="text", data="x")]

        async def fast():
            return [ReplySegment(type="text", data="y")]

        await mgr.submit(kind="ai_draw", work=slow, chat_id=CHAT)
        ack = await mgr.submit(kind="tts", work=fast, chat_id=CHAT)
        assert "在弄了" in ack
        gate.set()
        await asyncio.gather(*mgr._running.values(), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_no_chat_id_returns_fail(self, mgr, sent):
        # 显式清空路由 contextvar（防其他测试泄漏干扰）
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("")
        try:
            async def work():
                return [ReplySegment(type="text", data="x")]

            result = await mgr.submit(kind="ai_draw", work=work, chat_id="")
            assert result == "这次失败了，再试一次？"
            assert not mgr._running
        finally:
            current_chat_id.reset(token)

    @pytest.mark.asyncio
    async def test_timeout_fails(self, mgr, sent):
        async def work():
            await asyncio.sleep(10)

        await mgr.submit(kind="ai_draw", work=work, timeout=0.05, fail_text="超时了",
                         chat_id=CHAT)
        await asyncio.gather(*mgr._running.values(), return_exceptions=True)
        assert sent[0].segments[0].data == "超时了"

    @pytest.mark.asyncio
    async def test_slot_released_after_done(self, mgr, sent):
        async def work():
            return [ReplySegment(type="text", data="x")]

        await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert not mgr._running  # done_callback 已释放槽位
        ack = await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        assert "在弄了" in ack
        await asyncio.gather(*mgr._running.values())


class TestShutdown:
    @pytest.mark.asyncio
    async def test_cancel_all(self, mgr, sent):
        started = asyncio.Event()

        async def work():
            started.set()
            await asyncio.sleep(60)

        await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        await started.wait()
        await mgr.shutdown()
        assert not mgr._running
        assert not sent  # 取消的任务不发任何消息
