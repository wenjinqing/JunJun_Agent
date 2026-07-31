"""轻量规划循环（P0-12）测试：复杂度判定 / 计划生成解析 / 中间件注入。"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from junjun_agent.loop import plan_tracker as pt


async def _passthrough(r):
    return r


class TestDetectComplexity:
    def test_compound_with_connector(self):
        assert pt.detect_complexity("搜一下下周新番然后画张图发空间") is True

    def test_multi_actions_no_connector(self):
        assert pt.detect_complexity("帮我搜一下那个漫画，画个封面，再写段推荐语") is True

    def test_simple_question(self):
        assert pt.detect_complexity("你觉得明天天气怎么样") is False

    def test_short_text(self):
        assert pt.detect_complexity("搜一下") is False

    def test_single_action_with_connector(self):
        assert pt.detect_complexity("帮我搜一下这是什么") is False

    def test_empty(self):
        assert pt.detect_complexity("") is False


class _FakeResp:
    def __init__(self, content):
        self.content = content


class TestMakePlan:
    @pytest.mark.asyncio
    async def test_parse_numbered_list(self, monkeypatch):
        class _Model:
            async def ainvoke(self, msgs):
                return _FakeResp("1. 搜索新番信息\n2. 画一张配图\n3. 发到QQ空间")

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        steps = await pt.make_plan("搜新番然后画图发空间")
        assert steps == ["搜索新番信息", "画一张配图", "发到QQ空间"]

    @pytest.mark.asyncio
    async def test_direct_reply_degrades(self, monkeypatch):
        """LLM 判定单步任务 -> None（不注入清单）。"""
        class _Model:
            async def ainvoke(self, msgs):
                return _FakeResp("1. 直接回复")

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        assert await pt.make_plan("随便聊聊") is None

    @pytest.mark.asyncio
    async def test_llm_failure_degrades(self, monkeypatch):
        class _Model:
            async def ainvoke(self, msgs):
                raise RuntimeError("boom")

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        assert await pt.make_plan("搜一下然后画张图") is None

    @pytest.mark.asyncio
    async def test_max_steps_cap(self, monkeypatch):
        class _Model:
            async def ainvoke(self, msgs):
                return _FakeResp("\n".join(f"{i}. 步骤{i}" for i in range(1, 8)))

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        steps = await pt.make_plan("复杂任务")
        assert len(steps) == pt._MAX_STEPS


class _FakeRequest:
    """ModelRequest 替身：只需 messages + override。"""

    def __init__(self, messages):
        self.messages = messages

    def override(self, messages):
        return _FakeRequest(messages)


class TestPlanMiddleware:
    @pytest.mark.asyncio
    async def test_no_plan_passthrough(self):
        mw = pt.PlanMiddleware()
        token = pt.set_plan(None)
        req = _FakeRequest([HumanMessage(content="hi")])
        resp = await mw.awrap_model_call(req, _passthrough)
        pt.reset_plan(token)
        assert resp is req  # 未被修改

    @pytest.mark.asyncio
    async def test_injects_reminder_with_tool_count(self):
        mw = pt.PlanMiddleware()
        token = pt.set_plan(["搜索信息", "画图", "发空间"])
        req = _FakeRequest([
            HumanMessage(content="任务"),
            ToolMessage(content="结果1", tool_call_id="a"),
            ToolMessage(content="结果2", tool_call_id="b"),
        ])
        seen = {}

        async def _handler(r):
            seen["req"] = r
            return r

        await mw.awrap_model_call(req, _handler)
        pt.reset_plan(token)
        msgs = seen["req"].messages
        assert len(msgs) == 4  # 原 3 + 注入 1
        last = msgs[-1]
        assert isinstance(last, SystemMessage)
        assert "搜索信息" in last.content and "发空间" in last.content
        assert "已发起 2 次工具调用" in last.content

    @pytest.mark.asyncio
    async def test_single_step_plan_not_set(self):
        """不足 2 步的清单不注入。"""
        token = pt.set_plan(["就一步"])
        assert pt._current_plan.get() is None
        pt.reset_plan(token)


class TestReminderText:
    def test_reminder_format(self):
        text = pt._reminder(["步骤一", "步骤二"], 3)
        assert "1. 步骤一" in text and "2. 步骤二" in text
        assert "已发起 3 次工具调用" in text
