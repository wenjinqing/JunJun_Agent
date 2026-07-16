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
def _no_langfuse(monkeypatch):
    import junjun_llm.tracing as tr
    monkeypatch.setattr(tr, "get_callbacks", lambda: [])


@pytest.fixture(autouse=True)
def _no_freq_eval(monkeypatch):
    async def _noop(session):
        return None
    monkeypatch.setattr(proc_mod, "_maybe_adjust_frequency", _noop)


@pytest.fixture(autouse=True)
def _no_mood_eval(monkeypatch):
    from junjun_express.mood import mood_manager
    monkeypatch.setattr(mood_manager, "should_evaluate", lambda chat_id: False)


@pytest.fixture(autouse=True)
def _fast_postprocess(monkeypatch):
    """测试中关掉错别字与延迟（确定性）。"""
    from junjun_agent.postprocess import OutboundMessage

    def _plain(text, rand=None):
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
async def test_at_bot_bypasses_gate_and_replies(session, fake_gateway, monkeypatch):
    agent = _install_fake_agent(session, "在呢")

    async def _fail_gate(*a, **k):
        raise AssertionError("@ 旁路不应调 L2 gate")
    monkeypatch.setattr(proc_mod, "llm_gate", _fail_gate)

    await _add_and_handle(session, _meta("君君在吗", at_bot=True))
    assert len(fake_gateway.sent) == 1
    assert fake_gateway.sent[0].segments[0].data == "在呢"
    assert fake_gateway.sent[0].target_group_id == "999"
    assert agent.called == 1


@pytest.mark.asyncio
async def test_gate_no_reply_suppresses(session, fake_gateway, monkeypatch):
    agent = _install_fake_agent(session)

    from junjun_agent.funnel import GateDecision, L1Result
    async def _gate(*a, **k):
        return GateDecision.NO_REPLY
    monkeypatch.setattr(proc_mod, "llm_gate", _gate)
    monkeypatch.setattr(proc_mod, "rule_gate", lambda **k: L1Result.TO_GATE)

    await _add_and_handle(session, _meta("随便说说"))
    assert fake_gateway.sent == []
    assert agent.called == 0


@pytest.mark.asyncio
async def test_until_call_enters_silence_then_released_by_at(session, fake_gateway, monkeypatch):
    agent = _install_fake_agent(session, "我回来了")

    from junjun_agent.funnel import GateDecision, L1Result
    async def _gate(*a, **k):
        return GateDecision.NO_REPLY_UNTIL_CALL
    monkeypatch.setattr(proc_mod, "llm_gate", _gate)
    monkeypatch.setattr(proc_mod, "rule_gate", lambda **k: L1Result.TO_GATE)

    await _add_and_handle(session, _meta("君君闭嘴"))
    assert session.silenced_until_call is True

    from junjun_agent.funnel import rule_gate as real_gate
    monkeypatch.setattr(proc_mod, "rule_gate", real_gate)

    await _add_and_handle(session, _meta("路人甲说话"))
    assert session.silenced_until_call is True
    assert fake_gateway.sent == []

    await _add_and_handle(session, _meta("君君出来", at_bot=True))
    assert session.silenced_until_call is False
    assert len(fake_gateway.sent) == 1
    assert agent.called == 1


@pytest.mark.asyncio
async def test_memory_accumulates_even_when_dropped(session, fake_gateway, monkeypatch):
    _install_fake_agent(session)
    from junjun_agent.funnel import L1Result
    monkeypatch.setattr(proc_mod, "rule_gate", lambda **k: L1Result.DROP)

    await _add_and_handle(session, _meta("消息1"))
    await _add_and_handle(session, _meta("消息2"))
    assert len(session.memory.entries) == 2
    assert fake_gateway.sent == []


@pytest.mark.asyncio
async def test_multi_piece_reply_sends_multiple(session, fake_gateway, monkeypatch):
    """分条回复逐条发送，只有首条带引用。"""
    _install_fake_agent(session, "第一条。第二条。")
    from junjun_agent.postprocess import OutboundMessage

    def _two_pieces(text, rand=None):
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
        def dispatch(self, s, m, h):
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
