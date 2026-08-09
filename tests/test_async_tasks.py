"""异步任务管理器测试：提交/去重/超时/异常兜底/完成直发/优雅退出。"""

import asyncio
from types import SimpleNamespace

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


class TestAgentReport:
    """P0（2026-08-09 反馈闭环）：任务结局让 Agent 亲口播报，静态模板兜底。"""

    @pytest.fixture(autouse=True)
    def _clean_session(self):
        """假会话用完即清——全局 session_manager 泄漏会污染其他测试。"""
        from junjun_core.gateway.session_manager import get_session_manager
        get_session_manager().all_sessions().pop(CHAT, None)
        yield
        get_session_manager().all_sessions().pop(CHAT, None)

    @staticmethod
    def _install_fake_session(chat_id, model):
        """往全局 session_manager 塞一个带假模型的会话。"""
        from junjun_core.gateway.session_manager import ChatSession, get_session_manager
        session = ChatSession(chat_id, platform="qq", group_id=chat_id.split(":")[1])
        session.agent = SimpleNamespace(_model=model)
        get_session_manager().all_sessions()[chat_id] = session
        return session

    class _FakeModel:
        def __init__(self, text="画好啦，久等~", exc=None):
            self._text, self._exc = text, exc
            self.prompts = []

        async def ainvoke(self, messages):
            self.prompts.append(messages)
            if self._exc:
                raise self._exc
            return SimpleNamespace(content=self._text)

    @pytest.mark.asyncio
    async def test_success_voiced_by_agent(self, mgr, sent, monkeypatch):
        """成功结局：发的是 Agent 播报文本，不是模板池话术。"""
        model = self._FakeModel("画好啦，这次手感不错~")
        self._install_fake_session(CHAT, model)

        async def work():
            return [ReplySegment(type="image", data="http://x/1.png")]

        await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT, context="猫娘")
        await asyncio.gather(*mgr._running.values())
        segs = sent[0].segments
        assert segs[0].type == "text" and segs[0].data == "画好啦，这次手感不错~"
        assert segs[-1].type == "image"
        # 播报事件以 HumanMessage 一次性传入，带了任务主题
        human = model.prompts[0][-1].content
        assert "猫娘" in human and "完成" in human

    @pytest.mark.asyncio
    async def test_failure_voiced_with_reason(self, mgr, sent):
        """失败结局：播报里交代原因，不是干巴巴的模板。"""
        model = self._FakeModel("画砸了，接口超时了，换个描述再试？")
        self._install_fake_session(CHAT, model)

        async def work():
            return None

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了",
                         chat_id=CHAT, context="星空")
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画砸了，接口超时了，换个描述再试？"

    @pytest.mark.asyncio
    async def test_voice_failure_falls_back_to_template(self, mgr, sent):
        """播报 LLM 异常 -> 静态模板兜底（结局绝不能丢）。"""
        model = self._FakeModel(exc=RuntimeError("llm down"))
        self._install_fake_session(CHAT, model)

        async def work():
            return None

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画失败了"

    @pytest.mark.asyncio
    async def test_voice_echo_rejected(self, mgr, sent):
        """播报文本漏出「系统事件」标记 -> 视为坏输出，回退模板。"""
        model = self._FakeModel("[系统事件] 画图完成了")
        self._install_fake_session(CHAT, model)

        async def work():
            return None

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画失败了"

    @pytest.mark.asyncio
    async def test_switch_off_keeps_templates(self, mgr, sent, monkeypatch):
        """灰度开关关闭 -> 严格旧行为（不建会话也不调模型）。"""
        monkeypatch.setattr(TaskManager, "_agent_report_enabled", staticmethod(lambda: False))
        model = self._FakeModel("不该出现的话")
        self._install_fake_session(CHAT, model)

        async def work():
            return None

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画失败了"
        assert model.prompts == []

    @pytest.mark.asyncio
    async def test_no_session_falls_back(self, mgr, sent):
        """会话已淘汰（重启/长时间无消息）-> 模板兜底，不新建模型客户端。"""
        from junjun_core.gateway.session_manager import get_session_manager
        get_session_manager().all_sessions().pop(CHAT, None)

        async def work():
            return None

        await mgr.submit(kind="ai_draw", work=work, fail_text="画失败了", chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        assert sent[0].segments[0].data == "画失败了"

    @pytest.mark.asyncio
    async def test_outcome_event_not_in_stm_as_user(self, mgr, sent):
        """红线：播报事件不得以 user 身份进 STM（Hermes 幻影轮教训）；
        播报话术以 bot 身份记入（下轮被问「图呢」能答）。"""
        model = self._FakeModel("画好啦~")
        session = self._install_fake_session(CHAT, model)
        from junjun_memory.short_term import ShortTermMemory
        session.memory = ShortTermMemory()

        async def work():
            return [ReplySegment(type="image", data="http://x/1.png")]

        await mgr.submit(kind="ai_draw", work=work, chat_id=CHAT)
        await asyncio.gather(*mgr._running.values())
        user_entries = [e for e in session.memory.entries if e.role == "user"]
        bot_entries = [e for e in session.memory.entries if e.role == "bot"]
        assert not any("系统事件" in e.text for e in user_entries)
        assert any("画好啦~" in e.text for e in bot_entries)


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
