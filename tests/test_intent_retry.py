"""意图自检：强意图词命中但对应工具没调 -> 生成系统追问。

背景：弱模型把「帮我盯着xxx」当记忆任务只调 save_memory 或纯口头答应，
动作没生效用户却以为办好了。补救轮比换贵模型便宜一个数量级。
"""

from langchain_core.messages import AIMessage

import pytest

from junjun_agent.agent import _called_tool_names, _intent_nudge


def _ai_with_tools(*names):
    return AIMessage(content="", tool_calls=[{"name": n, "args": {}, "id": f"t{n}"} for n in names])


ALL_TOOLS = {"subscribe_updates", "unsubscribe", "set_reminder", "save_memory", "do_not_reply"}


class TestIntentNudge:
    def test_subscribe_intent_without_tool_call(self):
        """「帮我盯着p站16689973」只调了 save_memory -> 追问 subscribe_updates。"""
        msgs = [_ai_with_tools("save_memory")]
        nudge = _intent_nudge("@君君 帮我盯着p站作者16689973", msgs, ALL_TOOLS)
        assert nudge and "subscribe_updates" in nudge

    def test_no_nudge_when_tool_called(self):
        """已正确调用 subscribe_updates -> 不追问。"""
        msgs = [_ai_with_tools("subscribe_updates")]
        assert _intent_nudge("帮我盯着p站作者16689973", msgs, ALL_TOOLS) is None

    def test_no_nudge_when_tool_unavailable(self):
        """工具被掩码裁掉 -> 不追问（问了也调不了）。"""
        msgs = [_ai_with_tools("save_memory")]
        assert _intent_nudge("帮我盯着p站16689973", msgs, {"save_memory"}) is None

    def test_unsubscribe_takes_priority(self):
        """「取消订阅」含「订阅」——必须先命中 unsubscribe 规则。"""
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("取消订阅 3", msgs, ALL_TOOLS)
        assert nudge and "unsubscribe" in nudge

    def test_reminder_intent(self):
        msgs = [_ai_with_tools()]
        nudge = _intent_nudge("明天早上八点提醒我开会", msgs, ALL_TOOLS)
        assert nudge and "set_reminder" in nudge

    def test_research_intent_goes_background(self):
        """「调研」命中深度研究规则：只做了内联搜索 -> 追问 deep_research。"""
        tools = ALL_TOOLS | {"deep_research"}
        msgs = [_ai_with_tools("web_search")]
        nudge = _intent_nudge("帮我调研一下绝区零丹的攻略", msgs, tools)
        assert nudge and "deep_research" in nudge

    def test_research_no_nudge_when_submitted(self):
        tools = ALL_TOOLS | {"deep_research"}
        msgs = [_ai_with_tools("deep_research")]
        assert _intent_nudge("帮我调研一下绝区零丹的攻略", msgs, tools) is None

    def test_quick_lookup_no_nudge(self):
        """快查（查天气/搜快讯）不触发深度研究规则。"""
        tools = ALL_TOOLS | {"deep_research"}
        msgs = [_ai_with_tools("web_search")]
        assert _intent_nudge("帮我查下明天天气", msgs, tools) is None
        assert _intent_nudge("绝区零丹怎么配队", msgs, tools) is None

    def test_no_intent_no_nudge(self):
        msgs = [_ai_with_tools()]
        assert _intent_nudge("今天天气真好", msgs, ALL_TOOLS) is None
        assert _intent_nudge("", msgs, ALL_TOOLS) is None

    def test_called_tool_names(self):
        msgs = [_ai_with_tools("a", "b"), AIMessage(content="你好")]
        assert _called_tool_names(msgs) == {"a", "b"}


class TestAgentRebuild:
    @pytest.mark.asyncio
    async def test_agent_rebuilds_tools_each_round(self, monkeypatch):
        """回归（2026-08-01 trace）：agent 图必须每轮重建。

        曾经构造时绑死工具集——此时 memory 为空，关键词钉不住，
        run_background_task 等非 CORE 工具被裁后整个会话不可用，
        意图自检却按实时掩码追问 -> 模型被追问一个没绑定的工具。
        """
        import junjun_agent.agent as agent_mod
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel)
        from junjun_core.gateway.session_manager import ChatSession

        class _BindableFake(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        calls = []
        real_get_tools = agent_mod.get_tools

        def counting(session=None):
            calls.append(1)
            return real_get_tools(session)
        monkeypatch.setattr(agent_mod, "get_tools", counting)

        session = ChatSession("qq:1:private", "qq", user_id="1")
        agent = agent_mod.JunJunAgent(
            session, model=_BindableFake(responses=[AIMessage(content="好")]))
        assert calls == []  # 构造时不再绑工具
        await agent.process("甲: 你好")
        assert calls  # process 时按当前会话状态实时构建


class TestAddressedFallback:
    @pytest.mark.asyncio
    async def test_addressed_exception_gets_fallback(self, monkeypatch):
        """被 @ 时 agent 炸了（含 recursion limit）回实话，不装死。"""
        import junjun_agent.agent as agent_mod
        from junjun_core.gateway.session_manager import ChatSession

        class _BoomAgent:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("Recursion limit reached")

        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self: _BoomAgent())
        session = ChatSession("qq:1:private", "qq", user_id="1")
        agent = agent_mod.JunJunAgent(session, model=object())
        out = await agent.process("甲: 帮我盯着p站16689973", addressed=True)
        assert out and "没办成" in out

    @pytest.mark.asyncio
    async def test_unaddressed_exception_stays_silent(self, monkeypatch):
        """未被 @ 时保持沉默（不炸会话）。"""
        import junjun_agent.agent as agent_mod
        from junjun_core.gateway.session_manager import ChatSession

        class _BoomAgent:
            async def ainvoke(self, *a, **kw):
                raise RuntimeError("Recursion limit reached")

        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self: _BoomAgent())
        session = ChatSession("qq:1:group", "qq", group_id="1")
        agent = agent_mod.JunJunAgent(session, model=object())
        assert await agent.process("甲: 随便聊聊", addressed=False) is None
