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
