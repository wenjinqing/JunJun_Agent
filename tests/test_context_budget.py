"""背景上下文预算（2026-08-03 用户实测：Agent 只能看到 5-7 条消息）。

根因：processor 渲染 limit=30 条，但 agent.py 把背景硬砍到 10 行。
修复后默认 30 行（[chat] background_context_lines 可调）——
本测试钉住「30 条渲染进来就要 30 条都到模型」的语义。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from junjun_agent import agent as agent_mod


class _FakeGraph:
    captured = {}

    async def ainvoke(self, inp, config=None):
        _FakeGraph.captured["messages"] = inp["messages"]
        return {"messages": [AIMessage(content="好")]}


@pytest.fixture
def agent_env(monkeypatch):
    monkeypatch.setattr(agent_mod, "create_agent", lambda **kw: _FakeGraph())
    session = SimpleNamespace(chat_id="qq:1:group", is_group=True,
                              memory=SimpleNamespace(entries=[]))
    ag = agent_mod.JunJunAgent(session, model=SimpleNamespace())
    return ag


def _ctx(n=30):
    return "\n".join(f"群友{i}: 消息{i}" for i in range(n))


class TestBackgroundBudget:
    @pytest.mark.asyncio
    async def test_30_lines_all_reach_model(self, agent_env):
        """默认预算 30：第 1 条和第 18 条（旧预算 10 早被砍掉的）都要在背景里。"""
        await agent_env.process(_ctx(30), latest_text="消息29")
        msgs = _FakeGraph.captured["messages"]
        bg = next(m.content for m in msgs if str(m.content).startswith("[群聊背景"))
        assert "消息0" in bg and "消息18" in bg and "消息28" in bg
        latest = next(m.content for m in msgs if str(m.content).startswith("[你要回复的消息"))
        assert "消息29" in latest

    @pytest.mark.asyncio
    async def test_budget_configurable(self, agent_env):
        """[chat] background_context_lines 调小就只看最近的（省 token 出口）。"""
        import junjun_core.config.config as cfg_mod
        cfg_mod.global_config.raw["chat"]["background_context_lines"] = 5
        await agent_env.process(_ctx(30), latest_text="消息29")
        msgs = _FakeGraph.captured["messages"]
        bg = next(m.content for m in msgs if str(m.content).startswith("[群聊背景"))
        assert "消息28" in bg and "消息24" in bg
        assert "消息0" not in bg and "消息23" not in bg
