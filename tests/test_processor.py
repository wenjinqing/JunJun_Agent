"""processor 集成测（fake 模型走全链路，不打真实 API）。"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from junjun_core.gateway.router import InboundMeta
from junjun_core.gateway.session_manager import ChatSession
from junjun_agent import processor as proc_mod
from junjun_agent.processor import junjun_processor


def _meta(text: str, *, group="999", at_bot=False, is_self=False, user_id="111", msg_id="1"):
    return InboundMeta(
        text=text, user_id=user_id, nickname="甲",
        group_id=group, message_id=msg_id, at_bot=at_bot, is_self=is_self,
    )


@pytest.fixture
def session():
    return ChatSession("qq:999:group", "qq", group_id="999")


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch):
    import junjun_llm.tracing as tr
    monkeypatch.setattr(tr, "get_callbacks", lambda: [])


def _install_fake_agent(session, reply_text="哈喽"):
    """给 session 塞 fake agent + memory，绕过真实 LLM。"""
    from junjun_memory.short_term import ShortTermMemory

    class FakeAgent:
        def __init__(self):
            self.called = 0

        async def process(self, ctx, callbacks=None):
            self.called += 1
            return reply_text

    session.memory = ShortTermMemory()
    session.agent = FakeAgent()
    return session.agent


@pytest.mark.asyncio
async def test_self_message_silent(session):
    _install_fake_agent(session)
    assert await junjun_processor(session, _meta("x", is_self=True, at_bot=True)) is None


@pytest.mark.asyncio
async def test_at_bot_bypasses_gate_and_replies(session, monkeypatch):
    agent = _install_fake_agent(session, "在呢")

    async def _fail_gate(*a, **k):
        raise AssertionError("@ 旁路不应调 L2 gate")
    monkeypatch.setattr(proc_mod, "llm_gate", _fail_gate)

    reply = await junjun_processor(session, _meta("君君在吗", at_bot=True))
    assert reply is not None
    assert reply.segments[0].data == "在呢"
    assert reply.target_group_id == "999"
    assert agent.called == 1


@pytest.mark.asyncio
async def test_gate_no_reply_suppresses(session, monkeypatch):
    agent = _install_fake_agent(session)

    from junjun_agent.funnel import GateDecision
    async def _gate(*a, **k):
        return GateDecision.NO_REPLY
    monkeypatch.setattr(proc_mod, "llm_gate", _gate)
    # talk_value=0.9 有概率 DROP，锁定 rule_gate 结果稳定进 gate
    from junjun_agent.funnel import L1Result
    monkeypatch.setattr(proc_mod, "rule_gate", lambda **k: L1Result.TO_GATE)

    assert await junjun_processor(session, _meta("随便说说")) is None
    assert agent.called == 0


@pytest.mark.asyncio
async def test_until_call_enters_silence_then_released_by_at(session, monkeypatch):
    agent = _install_fake_agent(session, "我回来了")

    from junjun_agent.funnel import GateDecision, L1Result
    async def _gate(*a, **k):
        return GateDecision.NO_REPLY_UNTIL_CALL
    monkeypatch.setattr(proc_mod, "llm_gate", _gate)
    monkeypatch.setattr(proc_mod, "rule_gate", lambda **k: L1Result.TO_GATE)

    assert await junjun_processor(session, _meta("君君闭嘴")) is None
    assert session.silenced_until_call is True

    # 恢复真 rule_gate：沉默中普通消息 DROP，@ 解除
    from junjun_agent.funnel import rule_gate as real_gate
    monkeypatch.setattr(proc_mod, "rule_gate", real_gate)

    assert await junjun_processor(session, _meta("路人甲说话")) is None
    assert session.silenced_until_call is True

    reply = await junjun_processor(session, _meta("君君出来", at_bot=True))
    assert reply is not None
    assert session.silenced_until_call is False
    assert agent.called == 1


@pytest.mark.asyncio
async def test_memory_accumulates_even_when_silent(session, monkeypatch):
    _install_fake_agent(session)
    from junjun_agent.funnel import L1Result
    monkeypatch.setattr(proc_mod, "rule_gate", lambda **k: L1Result.DROP)

    await junjun_processor(session, _meta("消息1"))
    await junjun_processor(session, _meta("消息2"))
    assert len(session.memory.entries) == 2


class _BindableFakeChat(FakeMessagesListChatModel):
    """FakeMessagesListChatModel 不支持 bind_tools（create_agent 必需），补一个透传。"""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_real_agent_with_fake_llm_plain_reply():
    """JunJunAgent + fake 模型：验证 create_agent 全链路文本回复。"""
    from junjun_agent.agent import JunJunAgent
    session = ChatSession("qq:1:private", "qq", user_id="1")

    fake_llm = _BindableFakeChat(responses=[AIMessage(content="现在是晚上八点啦")])
    agent = JunJunAgent(session, model=fake_llm)
    text = await agent.process("甲: 几点了")
    assert text == "现在是晚上八点啦"


@pytest.mark.asyncio
async def test_real_agent_silence_via_tool_call():
    """agent 调 do_not_reply 工具 -> process 返回 None，哨兵文本不外泄。"""
    from junjun_agent.agent import JunJunAgent
    session = ChatSession("qq:1:private", "qq", user_id="1")

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "do_not_reply", "args": {"reason": "无关闲聊"}, "id": "tc1"}],
    )
    # 第二轮模型看到工具结果后输出的内容不应被外发
    fake_llm = _BindableFakeChat(responses=[tool_call_msg, AIMessage(content="（保持沉默）")])
    agent = JunJunAgent(session, model=fake_llm)
    assert await agent.process("甲: 随便聊聊") is None
