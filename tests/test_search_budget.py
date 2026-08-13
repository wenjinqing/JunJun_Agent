"""检索软刹车测试（2026-08-13 trace 4742c8fd 实锤：20 次搜索烧穿递归上限）。

SearchBudgetMiddleware：检索类工具（名字含 search）每轮软上限，
用完短路成「立即基于已有信息作答」的结构化文本；按尝试计数；
追问重试复用实例时预算延续；0=关闭。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from junjun_agent.loop.search_budget import SearchBudgetMiddleware, _budget_cfg
from junjun_core.config import get_global_config


def _req(name, call_id="c1"):
    return SimpleNamespace(tool_call={"name": name, "id": call_id})


class _Handler:
    """记录型假执行体：被调即记一次，返回正常 ToolMessage。"""

    def __init__(self):
        self.calls = []

    async def __call__(self, request):
        self.calls.append(request.tool_call["name"])
        return ToolMessage(content="搜索结果", tool_call_id=request.tool_call["id"],
                           name=request.tool_call["name"])


@pytest.fixture()
def budget2(monkeypatch):
    monkeypatch.setitem(get_global_config().raw, "agent", {"search_budget": 2})


class TestBudget:
    @pytest.mark.asyncio
    async def test_within_budget_passes(self, budget2):
        mw, h = SearchBudgetMiddleware(), _Handler()
        r1 = await mw.awrap_tool_call(_req("web_search"), h)
        r2 = await mw.awrap_tool_call(_req("mcp_tavily_search", "c2"), h)
        assert r1.content == "搜索结果" and r2.content == "搜索结果"
        assert h.calls == ["web_search", "mcp_tavily_search"]

    @pytest.mark.asyncio
    async def test_over_budget_short_circuits(self, budget2):
        """第 3 次搜索被短路：执行体没被调，返回「立即作答」结构化文本。"""
        mw, h = SearchBudgetMiddleware(), _Handler()
        await mw.awrap_tool_call(_req("web_search"), h)
        await mw.awrap_tool_call(_req("web_search", "c2"), h)
        r3 = await mw.awrap_tool_call(_req("mcp_search", "c3"), h)
        assert h.calls == ["web_search", "web_search"], "超限时不许真执行"
        assert "预算" in r3.content and "立即" in r3.content
        assert r3.tool_call_id == "c3" and r3.name == "mcp_search"  # 图状态一致性

    @pytest.mark.asyncio
    async def test_failed_attempts_count(self, budget2):
        """按尝试计：执行体内部失败（换乘救火的正是失败重试）同样烧预算。"""
        mw = SearchBudgetMiddleware()

        async def boom(request):
            raise ConnectionError("ECONNRESET")

        for i in range(2):
            with pytest.raises(ConnectionError):
                await mw.awrap_tool_call(_req("web_search", f"c{i}"), boom)
        r = await mw.awrap_tool_call(_req("web_search", "c9"), _Handler())
        assert "预算" in r.content

    @pytest.mark.asyncio
    async def test_non_search_tools_unaffected(self, budget2):
        """误判回归：非检索工具（画/表情/记忆）不受预算影响。"""
        mw, h = SearchBudgetMiddleware(), _Handler()
        for i in range(2):
            await mw.awrap_tool_call(_req("web_search", f"s{i}"), h)
        r = await mw.awrap_tool_call(_req("ai_draw", "d1"), h)
        assert r.content == "搜索结果"  # 非检索工具照常放行（内容为桩返回值）

    @pytest.mark.asyncio
    async def test_search_knowledge_counts_as_search(self, budget2):
        """名字含 search 即检索类——连刷本地知识库检索同样是病态，一起刹。"""
        mw, h = SearchBudgetMiddleware(), _Handler()
        await mw.awrap_tool_call(_req("web_search"), h)
        await mw.awrap_tool_call(_req("search_knowledge", "c2"), h)
        r = await mw.awrap_tool_call(_req("search_knowledge", "c3"), h)
        assert "预算" in r.content

    @pytest.mark.asyncio
    async def test_new_instance_resets(self, budget2):
        """每轮新建实例 = 预算按轮重置（_build_agent 每轮调用的语义）。"""
        mw1, h = SearchBudgetMiddleware(), _Handler()
        await mw1.awrap_tool_call(_req("web_search"), h)
        await mw1.awrap_tool_call(_req("web_search", "c2"), h)
        mw2 = SearchBudgetMiddleware()
        r = await mw2.awrap_tool_call(_req("web_search", "c3"), h)
        assert r.content == "搜索结果"

    @pytest.mark.asyncio
    async def test_zero_disables(self, monkeypatch):
        monkeypatch.setitem(get_global_config().raw, "agent", {"search_budget": 0})
        mw, h = SearchBudgetMiddleware(), _Handler()
        for i in range(10):
            r = await mw.awrap_tool_call(_req("web_search", f"c{i}"), h)
            assert r.content == "搜索结果"

    def test_bad_config_falls_back_to_default(self, monkeypatch):
        """配置写坏按默认 6（宁刹勿放）。"""
        monkeypatch.setitem(get_global_config().raw, "agent",
                            {"search_budget": "abc"})
        assert _budget_cfg() == 6

    def test_default_six_when_missing(self, monkeypatch):
        monkeypatch.setitem(get_global_config().raw, "agent", {})
        assert _budget_cfg() == 6
