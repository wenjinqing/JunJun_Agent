"""MCP 工具包装测试：任何调用失败降级为错误文本，绝不外抛炸掉 agent 轮次。"""

import asyncio

import pytest

from junjun_mcp_client.client import MCPManager


class _FakeTool:
    def __init__(self, name, coro):
        self.name = name
        self.coroutine = coro


def _wrap(coro):
    tool = _FakeTool("tavily_search", coro)
    return MCPManager()._wrap(tool)


class TestWrapErrorDegradation:
    @pytest.mark.asyncio
    async def test_generic_exception_becomes_error_text(self):
        """ToolException/网络错误 -> 错误文本结果，不抛异常。"""
        async def _boom(**kw):
            raise Exception("Tavily API error: read ECONNRESET")

        tool = _wrap(_boom)
        content, artifact = await tool.coroutine()
        assert artifact is None
        assert "失败" in content and "ECONNRESET" in content

    @pytest.mark.asyncio
    async def test_timeout_becomes_error_text(self):
        async def _slow(**kw):
            await asyncio.sleep(60)

        tool = _wrap(_slow)
        import junjun_mcp_client.client as c
        orig = c._TOOL_TIMEOUT
        c._TOOL_TIMEOUT = 0.05
        try:
            content, artifact = await tool.coroutine()
        finally:
            c._TOOL_TIMEOUT = orig
        assert "超时" in content

    @pytest.mark.asyncio
    async def test_success_passthrough(self):
        async def _ok(**kw):
            return ([{"type": "text", "text": "搜索结果"}], None)

        tool = _wrap(_ok)
        content, artifact = await tool.coroutine()
        assert content == "搜索结果"

    @pytest.mark.asyncio
    async def test_name_prefixed(self):
        async def _ok(**kw):
            return "x", None

        tool = _wrap(_ok)
        assert tool.name == "mcp_tavily_search"
