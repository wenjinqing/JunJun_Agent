"""Skill 注册表单测。"""

import pytest
from langchain_core.tools import tool

from junjun_skills import registry


@tool
def dummy_skill(x: str) -> str:
    """测试用工具。

    Args:
        x: 输入
    """
    return x


class TestRegistry:
    def test_register_and_get(self):
        registry.register(dummy_skill)
        assert dummy_skill in registry.get_tools()

    def test_duplicate_name_raises(self):
        registry.register(dummy_skill)
        with pytest.raises(ValueError, match="重名"):
            registry.register(dummy_skill)

    def test_availability_filter(self):
        class FakeSession:
            group_id = "999"

        registry.register(dummy_skill, available_for=lambda s: s.group_id == "888")
        assert dummy_skill not in registry.get_tools(FakeSession())
        assert dummy_skill in registry.get_tools()  # session=None 全量


class TestBuiltinSchema:
    def test_builtin_tools_have_valid_schema(self):
        registry.load_builtin()
        tools = {t.name: t for t in registry.get_tools()}
        assert "get_time" in tools
        assert "do_not_reply" in tools
        # do_not_reply 必须有 reason 参数（function calling schema 合法）
        assert "reason" in tools["do_not_reply"].args

    def test_do_not_reply_returns_confirmation(self):
        from junjun_skills.builtin.do_not_reply import do_not_reply, SILENCE_TOOL_NAME
        assert do_not_reply.name == SILENCE_TOOL_NAME
        out = do_not_reply.invoke({"reason": "测试"})
        assert "沉默" in out


# ---------------------------------------------------------------- P0-13 错误结构化

class TestClassifyError:
    """异常 -> 错误类别（网络/参数/权限/限流/未知）。"""

    @staticmethod
    def _http_error(code: int):
        import httpx
        req = httpx.Request("GET", "http://x.test/")
        resp = httpx.Response(code, request=req)
        return httpx.HTTPStatusError("err", request=req, response=resp)

    def test_network(self):
        import asyncio

        import httpx
        assert registry._classify_error(httpx.ConnectError("refused")) == "网络"
        assert registry._classify_error(httpx.ReadTimeout("slow")) == "网络"
        assert registry._classify_error(asyncio.TimeoutError()) == "网络"
        assert registry._classify_error(ConnectionError("reset")) == "网络"

    def test_http_status(self):
        assert registry._classify_error(self._http_error(429)) == "限流"
        assert registry._classify_error(self._http_error(403)) == "权限"
        assert registry._classify_error(self._http_error(401)) == "权限"
        assert registry._classify_error(self._http_error(500)) == "网络"

    def test_permission_not_swallowed_by_oserror(self):
        """PermissionError 是 OSError 子类——必须先判权限，否则被网络类吞掉。"""
        assert registry._classify_error(PermissionError("denied")) == "权限"

    def test_params(self):
        assert registry._classify_error(ValueError("bad")) == "参数"
        assert registry._classify_error(TypeError("bad")) == "参数"
        assert registry._classify_error(KeyError("bad")) == "参数"

    def test_unknown(self):
        assert registry._classify_error(RuntimeError("boom")) == "未知"


class TestErrorFeedbackWrap:
    """register 统一包错误层：逃逸异常 -> [TOOL_ERROR kind=...] 文本，不再抛出。"""

    def test_async_tool_error_structured(self):
        import asyncio

        import httpx

        @tool
        async def flaky_async(x: str) -> str:
            """会挂的异步工具。

            Args:
                x: 输入
            """
            raise httpx.ConnectError("connection refused")

        registry.register(flaky_async)
        out = asyncio.run(flaky_async.ainvoke({"x": "1"}))
        assert out.startswith("[TOOL_ERROR kind=网络]")
        assert "flaky_async" in out

    def test_sync_tool_error_structured(self):
        @tool
        def flaky_sync(x: str) -> str:
            """会挂的同步工具。

            Args:
                x: 输入
            """
            raise ValueError("bad arg")

        registry.register(flaky_sync)
        out = flaky_sync.invoke({"x": "1"})
        assert out.startswith("[TOOL_ERROR kind=参数]")

    def test_normal_return_passthrough(self):
        registry.register(dummy_skill)
        assert dummy_skill.invoke({"x": "ok"}) == "ok"

    def test_error_text_truncated(self):
        @tool
        def noisy(x: str) -> str:
            """长报错工具。

            Args:
                x: 输入
            """
            raise RuntimeError("x" * 500)

        registry.register(noisy)
        out = noisy.invoke({"x": "1"})
        assert out.startswith("[TOOL_ERROR kind=未知]")
        assert len(out) < 250  # 前缀 + 类型名 + 截断 150，不灌爆上下文

    def test_admin_gate_inside_error_wrap(self, monkeypatch):
        """admin 门 + 错误层共存：非管理员仍拿到权限拒绝文本（不是 TOOL_ERROR）。"""
        monkeypatch.setenv("ADMIN_QQ", "10001")
        from junjun_core.security import set_caller
        set_caller("12345", at_bot=True, is_group=True)  # 显式非管理员（防全量跑的上下文泄漏）

        @tool
        def admin_thing(x: str) -> str:
            """管理员工具。

            Args:
                x: 输入
            """
            return "done"

        registry.register(admin_thing, admin_only=True)
        out = admin_thing.invoke({"x": "1"})
        set_caller("", at_bot=False, is_group=True)
        # 非管理员 -> 权限门先触发，错误层不影响
        assert "权限不足" in out


class TestFallbackMapPrompt:
    """换乘地图进 system prompt（persona rules）。"""

    def test_fallback_rules_in_prompt(self, _fake_bot_config):
        from junjun_agent.persona import build_system_prompt
        prompt = build_system_prompt(is_group=True)
        assert "[TOOL_ERROR kind=" in prompt
        assert "网络" in prompt and "限流" in prompt and "权限" in prompt
        assert "最多重试一次" in prompt
