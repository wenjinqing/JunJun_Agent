"""工具使用统计测试（2026-08-13 审查 P1）：record 落库 / registry 单点挂钩 /
日报可见 / 清理纳管。全程 tmp 库 + bind_ctx（CLAUDE.md 硬约束）。
"""

import time

import pytest
from langchain_core.tools import tool
from peewee import SqliteDatabase


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch):
    import junjun_core.database.models as m
    test_db = SqliteDatabase(":memory:")
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db


class TestRecord:
    def test_record_writes_row(self):
        from junjun_core.database import ToolUsage
        from junjun_skills import usage
        usage.record("web_search", True, chat_id="qq:1:group")
        row = ToolUsage.get()
        assert row.tool == "web_search" and row.ok is True
        assert row.chat_id == "qq:1:group"

    def test_record_never_raises(self, monkeypatch):
        """统计故障绝不挡工具主路径。"""
        from junjun_core.database import db_writer
        from junjun_skills import usage
        monkeypatch.setattr(db_writer, "submit",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("炸")))
        usage.record("web_search", True)  # 不抛

    def test_failure_kind_recorded(self):
        from junjun_core.database import ToolUsage
        from junjun_skills import usage
        usage.record("ai_draw", False, "限流", "qq:1:group")
        row = ToolUsage.get(ToolUsage.ok == False)  # noqa: E712
        assert row.error_kind == "限流"


class TestRegistryHook:
    """registry 错误包装层是唯一挂钩点：成功/失败都记账，带会话。"""

    def test_success_and_failure_counted(self):
        from junjun_core.database import ToolUsage
        from junjun_skills import registry
        from junjun_skills.builtin.memory_skills import current_chat_id

        @tool
        def ok_tool(x: str) -> str:
            """正常工具。

            Args:
                x: 输入
            """
            return "好"

        @tool
        def bad_tool(x: str) -> str:
            """必挂工具。

            Args:
                x: 输入
            """
            raise ValueError("坏")

        registry.register(ok_tool)
        registry.register(bad_tool)
        token = current_chat_id.set("qq:stat:group")
        try:
            ok_tool.invoke({"x": "1"})
            bad_tool.invoke({"x": "1"})
        finally:
            current_chat_id.reset(token)
        rows = {r.tool: r for r in ToolUsage.select()}
        assert rows["ok_tool"].ok is True
        assert rows["bad_tool"].ok is False
        assert rows["bad_tool"].error_kind == "参数"
        assert rows["ok_tool"].chat_id == "qq:stat:group"


class TestDailyReportVisibility:
    @pytest.mark.asyncio
    async def test_report_includes_tool_top(self, monkeypatch):
        """统计只入库没人看等于没统计——日报必须带工具 Top5。"""
        from junjun_core import alerting
        from junjun_core.database import LLMUsage, ToolUsage
        now = time.time()
        LLMUsage.create(time=now, request_type="agent",
                        prompt_tokens=100, completion_tokens=50)
        ToolUsage.create(time=now, tool="web_search", ok=True)
        ToolUsage.create(time=now, tool="web_search", ok=False, error_kind="网络")
        ToolUsage.create(time=now, tool="ai_draw", ok=True)
        sent = []

        async def _notify(text):
            sent.append(text)
            return True

        monkeypatch.setattr(alerting, "_safe_notify", _notify)
        await alerting.daily_usage_report()
        assert sent and "工具 Top5" in sent[0]
        assert "web_search×2(败1)" in sent[0]

    def test_cleanup_purges_old_usage(self):
        """ToolUsage 随 LLMUsage 同窗清理（表必须有界）。"""
        from junjun_core.database import ToolUsage
        from junjun_core.database.cleanup import _do_cleanup
        now = time.time()
        ToolUsage.create(time=now - 90 * 86400, tool="old_tool", ok=True)
        ToolUsage.create(time=now, tool="new_tool", ok=True)
        _do_cleanup(cutoff=now - 60 * 86400, msg_cutoff=0.0)
        left = {r.tool for r in ToolUsage.select()}
        assert left == {"new_tool"}
