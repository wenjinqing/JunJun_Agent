"""processor 集成测（fake 模型走全链路，不打真实 API）。

阶段 3 起 junjun_processor 只入队，核心决策在 _handle（本文件直接测 _handle）；
发送走 gateway.send_reply，用 fake gateway 捕获。
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from junjun_core.gateway.router import InboundMeta
from junjun_core.gateway.session_manager import ChatSession
from junjun_agent import processor as proc_mod
from junjun_agent.processor import _handle, junjun_processor


def _meta(text: str, *, group="999", at_bot=False, is_self=False, user_id="111", msg_id="1"):
    return InboundMeta(
        text=text, user_id=user_id, nickname="甲",
        group_id=group, message_id=msg_id, at_bot=at_bot, is_self=is_self,
    )


class FakeGateway:
    def __init__(self):
        self.sent = []

    async def send_reply(self, reply):
        self.sent.append(reply)


@pytest.fixture
def session():
    return ChatSession("qq:999:group", "qq", group_id="999")


@pytest.fixture
def fake_gateway(monkeypatch):
    gw = FakeGateway()
    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "_gateway", gw)
    return gw


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch):
    """消息管线会落库（_store_inbound/_store_outbound/intimacy）——绝不许写
    生产库（2026-08-06 第三次污染事故实锤：本文件与 test_queue_predecision
    把「第一条。第二条。」「私聊回复」等百余行写进了 data/junjun.db）。"""
    import junjun_core.database.models as m
    from peewee import SqliteDatabase
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch):
    import junjun_llm.tracing as tr
    monkeypatch.setattr(tr, "get_callbacks", lambda: [])


@pytest.fixture(autouse=True)
def _no_mood_eval(monkeypatch):
    from junjun_express.mood import mood_manager
    monkeypatch.setattr(mood_manager, "should_evaluate", lambda chat_id: False)


@pytest.fixture(autouse=True)
def _fast_postprocess(monkeypatch):
    """测试中关掉错别字与延迟（确定性）。"""
    from junjun_agent.postprocess import OutboundMessage

    def _plain(text, rand=None, incoming=''):
        return [OutboundMessage(text=text, delay=0.0)]
    monkeypatch.setattr(proc_mod, "process_response", _plain)


def _install_fake_agent(session, reply_text="哈喽"):
    from junjun_memory.short_term import ShortTermMemory

    class FakeAgent:
        def __init__(self):
            self.called = 0

        async def process(self, ctx, callbacks=None, **kw):
            self.called += 1
            return reply_text

    session.memory = ShortTermMemory()
    session.agent = FakeAgent()
    return session.agent


def _add_and_handle(session, meta):
    """模拟 junjun_processor 的入队前动作 + _handle。"""
    session.memory.add_user(meta.text, meta.nickname, user_id=meta.user_id or "",
                            message_id=meta.message_id, at_bot=meta.at_bot)
    return _handle(session, meta)


@pytest.mark.asyncio
async def test_self_message_silent(session, fake_gateway):
    _install_fake_agent(session)
    await _add_and_handle(session, _meta("x", is_self=True, at_bot=True))
    assert fake_gateway.sent == []


@pytest.mark.asyncio
async def test_at_bot_replies(session, fake_gateway):
    """@ 君君 -> 进思考回复。"""
    agent = _install_fake_agent(session, "在呢")

    await _add_and_handle(session, _meta("君君在吗", at_bot=True))
    assert len(fake_gateway.sent) == 1
    assert fake_gateway.sent[0].segments[0].data == "在呢"
    assert fake_gateway.sent[0].target_group_id == "999"
    assert agent.called == 1


@pytest.mark.asyncio
async def test_group_not_addressed_silent(session, fake_gateway):
    """群聊非 @/直呼 -> 沉默（无 planner，直接不处理）。"""
    agent = _install_fake_agent(session)

    await _add_and_handle(session, _meta("随便说说"))
    assert fake_gateway.sent == []
    assert agent.called == 0


@pytest.mark.asyncio
async def test_private_direct_replies(fake_gateway):
    """私聊直通——不需要 @/直呼，直接进决策。"""
    session = ChatSession("qq:12345:private", "qq", user_id="12345")
    agent = _install_fake_agent(session, "私聊回复")

    await _add_and_handle(session, _meta("你好", group=None))
    assert len(fake_gateway.sent) == 1
    assert agent.called == 1


@pytest.mark.asyncio
async def test_nickname_call_replies(session, fake_gateway):
    """直呼名字 -> 进思考回复。"""
    agent = _install_fake_agent(session, "叫我干嘛")

    await _add_and_handle(session, _meta("君君出来玩"))
    assert len(fake_gateway.sent) == 1
    assert agent.called == 1


@pytest.mark.asyncio
async def test_silenced_until_call_released_by_at(session, fake_gateway):
    """沉默模式中仅被呼唤解除。"""
    agent = _install_fake_agent(session, "我回来了")
    session.silenced_until_call = True

    await _add_and_handle(session, _meta("路人甲说话"))
    assert session.silenced_until_call is True
    assert fake_gateway.sent == []

    await _add_and_handle(session, _meta("君君出来", at_bot=True))
    assert session.silenced_until_call is False
    assert len(fake_gateway.sent) == 1
    assert agent.called == 1


@pytest.mark.asyncio
async def test_memory_accumulates_even_when_silent(session, fake_gateway):
    """沉默时记忆照常积累（上下文完整性）。"""
    _install_fake_agent(session)

    await _add_and_handle(session, _meta("消息1"))
    await _add_and_handle(session, _meta("消息2"))
    assert len(session.memory.entries) == 2
    assert fake_gateway.sent == []


@pytest.mark.asyncio
async def test_multi_piece_reply_sends_multiple(session, fake_gateway, monkeypatch):
    """分条回复逐条发送，只有首条带引用。"""
    _install_fake_agent(session, "第一条。第二条。")
    from junjun_agent.postprocess import OutboundMessage

    def _two_pieces(text, rand=None, incoming=''):
        return [OutboundMessage("第一条", 0.0), OutboundMessage("第二条", 0.0)]
    monkeypatch.setattr(proc_mod, "process_response", _two_pieces)
    monkeypatch.setattr(proc_mod, "_quote_message_id", lambda s, m: "42")

    await _add_and_handle(session, _meta("君君说个长的", at_bot=True))
    assert len(fake_gateway.sent) == 2
    assert fake_gateway.sent[0].reply_to_message_id == "42"
    assert fake_gateway.sent[1].reply_to_message_id is None


@pytest.mark.asyncio
async def test_processor_entry_enqueues(session, monkeypatch):
    """junjun_processor 入口：记忆即时写入 + 投递队列。"""
    _install_fake_agent(session)
    calls = []

    class FakeQueues:
        def dispatch(self, s, m, h, **_kw):
            calls.append((s, m))
    import junjun_agent.funnel.session_queue as sq
    monkeypatch.setattr(sq, "session_queues", FakeQueues())

    result = await junjun_processor(session, _meta("hello"))
    assert result is None
    assert len(session.memory.entries) == 1
    assert len(calls) == 1


class _BindableFakeChat(FakeMessagesListChatModel):
    """FakeMessagesListChatModel 不支持 bind_tools（create_agent 必需），补一个透传。"""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_real_agent_with_fake_llm_plain_reply():
    from junjun_agent.agent import JunJunAgent
    session = ChatSession("qq:1:private", "qq", user_id="1")

    fake_llm = _BindableFakeChat(responses=[AIMessage(content="现在是晚上八点啦")])
    agent = JunJunAgent(session, model=fake_llm)
    text = await agent.process("甲: 几点了")
    assert text == "现在是晚上八点啦"


@pytest.mark.asyncio
async def test_real_agent_silence_via_tool_call():
    from junjun_agent.agent import JunJunAgent
    session = ChatSession("qq:1:private", "qq", user_id="1")

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "do_not_reply", "args": {"reason": "无关闲聊"}, "id": "tc1"}],
    )
    fake_llm = _BindableFakeChat(responses=[tool_call_msg, AIMessage(content="（保持沉默）")])
    agent = JunJunAgent(session, model=fake_llm)
    assert await agent.process("甲: 随便聊聊") is None


@pytest.mark.asyncio
async def test_keyword_reaction_injected():
    """keyword_reaction 命中时注入 system prompt。"""
    from junjun_agent.persona import build_system_prompt, match_keyword_rules

    hits = match_keyword_rules("你是不是机器人啊")
    assert hits, "关键词应命中"
    prompt = build_system_prompt(is_group=True, latest_text="你是不是机器人啊")
    assert "特别注意" in prompt


class _RecSpan:
    """记录型假 span（对齐 test_task_kernel 的桩）。"""

    def __init__(self, sink, **kw):
        self.kw = kw
        self.updates = []
        self._sink = sink

    def __enter__(self):
        self._sink.append(self)
        return self

    def __exit__(self, *e):
        return False

    def update(self, **kw):
        self.updates.append(kw)


@pytest.mark.asyncio
async def test_honesty_guard_interception_traced(session, fake_gateway, monkeypatch):
    """2026-08-06 实锤：HonestyGuard 替换稿曾被发出但 trace 里是原文——
    校验必须挪进 span，实发稿 + 原文 + 拦截理由全部进 trace。"""
    from types import SimpleNamespace
    import junjun_core.observability as obs
    import junjun_core.config.config as cfg_mod

    spans = []
    monkeypatch.setattr(obs, "lf", SimpleNamespace(
        start_span=lambda **kw: _RecSpan(spans, **kw), enabled=True))
    cfg = cfg_mod.get_global_config()
    monkeypatch.setitem(cfg.raw, "honesty_guard", {"enable": True})

    _install_fake_agent(session, "画好了，等下发给你")  # 声称但没调工具
    await _add_and_handle(session, _meta("君君帮我画只猫", at_bot=True))

    assert len(fake_gateway.sent) == 1
    sent_text = fake_gateway.sent[0].segments[0].data
    assert "不能骗你" in sent_text          # 实发的是替换稿
    assert "画好" in sent_text              # 引用实际说出口的短语
    assert "[" not in sent_text             # 没有 regex 源码糊脸

    span = next(s for s in spans if s.kw["name"].startswith("agent."))
    final = span.updates[-1]["output"]
    assert final["reply"][:20] == sent_text[:20]     # trace 里就是实发稿
    hg = final["honesty_guard"]
    assert hg["intercepted"] is True
    assert "画好了，等下发给你" in hg["original"]    # 原文也可查
    assert any("ai_draw" in i for i in hg["issues"])
