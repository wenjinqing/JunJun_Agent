"""str 参数宽松化：模型把数字 ID 传成 JSON number 时服务端转字符串（2026-08-01 trace）。

Qwen 系模型常传 target=16689973（int），pydantic v2 str 字段拒收，
模型原样重试烧穿迭代上限。registry 注册时统一 BeforeValidator 宽松化。
"""

import pytest
from langchain_core.tools import tool

from junjun_skills import registry


@tool
def _demo_tool(name: str, uid: str, count: int) -> str:
    """测试工具。

    Args:
        name: 名字
        uid: 数字 ID（应为字符串）
        count: 次数
    """
    return f"{name}|{uid}|{type(uid).__name__}|{count}"


@pytest.fixture
def registered():
    registry.clear()
    registry.register(_demo_tool)
    yield registry._registry["_demo_tool"]
    registry.clear()


class TestRelaxStrArgs:
    def test_int_coerced_to_str(self, registered):
        """int 传参 -> str 到达函数体。"""
        out = registered.invoke({"name": "x", "uid": 16689973, "count": 3})
        assert out == "x|16689973|str|3"

    def test_str_passthrough(self, registered):
        out = registered.invoke({"name": "x", "uid": "16689973", "count": 3})
        assert out == "x|16689973|str|3"

    def test_int_field_untouched(self, registered):
        """int 字段不受宽松化影响（str 传 count 仍报错或按 pydantic  coercion）。"""
        schema = registered.args_schema.model_json_schema()
        assert schema["properties"]["count"]["type"] == "integer"

    def test_model_facing_schema_unchanged(self, registered):
        """模型侧 JSON schema 仍声明 string（宽松只在服务端）。"""
        schema = registered.args_schema.model_json_schema()
        assert schema["properties"]["uid"]["type"] == "string"

    def test_tool_description_and_required_preserved(self, registered):
        """工具 docstring（模型理解工具的命根子）与参数必填性不受重建影响。"""
        assert "测试工具" in registered.description
        schema = registered.args_schema.model_json_schema()
        assert set(schema["required"]) == {"name", "uid", "count"}
