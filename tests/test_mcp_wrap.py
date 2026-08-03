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


class TestDeterministicNoRetry:
    """确定性 MCP 失败（-32602 参数错等）不重试——重试只是白等还刷服务端报错
    （2026-08-03 实战：BV 号格式传错被重试 3 次）。"""

    @pytest.mark.asyncio
    async def test_invalid_params_not_retried(self):
        calls = []

        async def _bad_args(**kw):
            calls.append(1)
            raise Exception("McpError: MCP error -32602: 无效的视频ID格式，"
                            "请提供BV号（如：BV1xx411c7mD）")

        tool = _wrap(_bad_args)
        content, artifact = await tool.coroutine()
        assert len(calls) == 1, "参数错不该重试"
        assert artifact is None
        assert "工具拒绝了这次调用" in content
        assert "BV" in content  # 服务端的正确用法提示原样喂回（模型自我纠正）

    @pytest.mark.asyncio
    async def test_method_not_found_not_retried(self):
        calls = []

        async def _no_method(**kw):
            calls.append(1)
            raise Exception("McpError: MCP error -32601: Method not found")

        tool = _wrap(_no_method)
        content, _ = await tool.coroutine()
        assert len(calls) == 1
        assert "工具拒绝了这次调用" in content

    @pytest.mark.asyncio
    async def test_transient_still_retried(self):
        """瞬态错误（连接重置）保持原重试语义。"""
        calls = []

        async def _flaky(**kw):
            calls.append(1)
            raise ConnectionError("read ECONNRESET")

        tool = _wrap(_flaky)
        content, _ = await tool.coroutine()
        assert len(calls) == 3
        assert "失败" in content

    @pytest.mark.asyncio
    async def test_internal_error_still_retried(self):
        """-32603 服务端内部错误不在确定性名单（可能瞬态），照常重试。"""
        calls = []

        async def _internal(**kw):
            calls.append(1)
            raise Exception("McpError: MCP error -32603: Internal error")

        tool = _wrap(_internal)
        await tool.coroutine()
        assert len(calls) == 3

