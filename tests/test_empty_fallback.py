"""必回场景空输出兜底测试（2026-08-06 生产实锤：@必回消息模型返回空 content——
疑似合规自我审查——全程无日志只剩一条「L3 沉默」，用户被晾）。

与 GraphRecursionError 同原则：装死是最差的回复。先追问一次（_EMPTY_NUDGE），
仍空则回一句诚实的人话；非必回场景照旧沉默（不打扰群）。
"""

import pytest
from langchain_core.messages import AIMessage


def _session_with_memory():
    from junjun_core.gateway.session_manager import ChatSession
    from junjun_memory.short_term import ShortTermMemory
    session = ChatSession("qq:1:private", "qq", user_id="1")
    session.memory = ShortTermMemory()
    return session


class _ScriptedAgent:
    """按脚本逐次返回文本的假 agent 图（content 可为空串）。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def ainvoke(self, params, config=None):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return {"messages": [AIMessage(content=out)]}


class TestAgentEmptyFallback:
    @pytest.mark.asyncio
    async def test_addressed_empty_retries_with_nudge(self, monkeypatch):
        """首轮空输出 + 被@ -> 追问一次，第二轮说出话 -> 发第二轮。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["", "刚走神了，你刚说啥来着？"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False, **_kw: scripted)
        agent = agent_mod.JunJunAgent(_session_with_memory(), model=object())
        out = await agent.process("甲: @君君 说话", addressed=True)
        assert out == "刚走神了，你刚说啥来着？"
        assert scripted.calls == 2  # 确实追问了一轮

    @pytest.mark.asyncio
    async def test_addressed_still_empty_honest_fallback(self, monkeypatch):
        """追问后仍空 -> 诚实兜底句，绝不装死（L3 沉默事故修复）。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["", ""])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False, **_kw: scripted)
        agent = agent_mod.JunJunAgent(_session_with_memory(), model=object())
        out = await agent.process("甲: @君君 说话", addressed=True)
        assert out == "……抱歉，刚才那句我死活没组织出来。换个说法再问我一次？"
        assert scripted.calls == 2

    @pytest.mark.asyncio
    async def test_unaddressed_empty_stays_silent(self, monkeypatch):
        """非必回场景空输出：不追问不兜底，照常沉默（不打扰群）。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent([""])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False, **_kw: scripted)
        agent = agent_mod.JunJunAgent(_session_with_memory(), model=object())
        assert await agent.process("甲: 说点什么", addressed=False) is None
        assert scripted.calls == 1  # 不追问

    @pytest.mark.asyncio
    async def test_nonempty_first_round_no_retry(self, monkeypatch):
        """首轮正常输出：不触发空输出追问，一轮结束。"""
        import junjun_agent.agent as agent_mod

        scripted = _ScriptedAgent(["今晚月亮真圆啊"])
        monkeypatch.setattr(agent_mod.JunJunAgent, "_build_agent",
                            lambda self, full=False, **_kw: scripted)
        agent = agent_mod.JunJunAgent(_session_with_memory(), model=object())
        out = await agent.process("甲: @君君 看月亮", addressed=True)
        assert out == "今晚月亮真圆啊"
        assert scripted.calls == 1


class TestCompactBudgetMetrics:
    """langfuse v4 propagated attribute 限 200 字符（超限整条丢弃还刷警告）：
    预算指标压成紧凑摘要，核心字段保留。"""

    def test_empty_metrics(self):
        from junjun_agent.agent import _compact_budget_metrics
        assert _compact_budget_metrics({}) == ""

    def test_core_fields_kept(self):
        from junjun_agent.agent import _compact_budget_metrics
        s = _compact_budget_metrics({
            "kept_total_tokens": 4200, "max_total_tokens": 6000,
            "evicted_total_tokens": 800, "evicted_names": ["relation", "background"],
            "block_sizes": {"core": 2000, "mood": 100},
        })
        assert "kept=4200/6000" in s
        assert "evict=800[relation,background]" in s
        assert "core:2000" in s
        assert len(s) <= 200

    def test_oversized_truncated_to_200(self):
        """block 再多也不超 200（超限会被 langfuse 整条丢弃，不如主动截断）。"""
        from junjun_agent.agent import _compact_budget_metrics
        s = _compact_budget_metrics({
            "kept_total_tokens": 6000, "max_total_tokens": 6000,
            "evicted_total_tokens": 0, "evicted_names": [],
            "block_sizes": {f"block_{i:03d}": 1000 + i for i in range(50)},
        })
        assert len(s) <= 200
        assert s.startswith("kept=6000/6000")
