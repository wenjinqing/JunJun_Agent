"""阶段 6 单测：MCP 配置解析 / 插件加载器 / relationship server 工具。"""

import json

import pytest
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


class TestMCPConfig:
    def test_load_config_replaces_repo_root(self, tmp_path, monkeypatch):
        import junjun_mcp_client.client as c
        cfg = tmp_path / "mcp_servers.toml"
        cfg.write_text(
            '[servers.rel]\nenable = true\ncommand = "python"\n'
            'args = ["-m", "x"]\ncwd = "${REPO_ROOT}"\n', encoding="utf-8")
        monkeypatch.setattr(c, "MCP_CONFIG", cfg)
        servers = c.load_server_configs()
        assert "rel" in servers
        assert "${REPO_ROOT}" not in servers["rel"]["cwd"]
        assert servers["rel"]["transport"] == "stdio"

    def test_disabled_server_skipped(self, tmp_path, monkeypatch):
        import junjun_mcp_client.client as c
        cfg = tmp_path / "mcp_servers.toml"
        cfg.write_text('[servers.off]\nenable = false\ncommand = "x"\n', encoding="utf-8")
        monkeypatch.setattr(c, "MCP_CONFIG", cfg)
        assert c.load_server_configs() == {}

    def test_missing_config_empty(self, tmp_path, monkeypatch):
        import junjun_mcp_client.client as c
        monkeypatch.setattr(c, "MCP_CONFIG", tmp_path / "nope.toml")
        assert c.load_server_configs() == {}


class TestPluginLoader:
    def test_load_valid_plugin(self, tmp_path, monkeypatch):
        import junjun_skills.plugin_loader as pl
        # 构造临时插件
        plug = tmp_path / "demo"
        plug.mkdir()
        (plug / "_manifest.json").write_text(json.dumps({
            "name": "demo", "version": "1.0",
            "module": "tests._demo_plugin_mod", "tools_attr": "TOOLS",
        }), encoding="utf-8")
        monkeypatch.setattr(pl, "PLUGINS_DIR", tmp_path)

        import sys, types
        from langchain_core.tools import tool

        @tool
        def demo_tool() -> str:
            """demo。"""
            return "ok"
        mod = types.ModuleType("tests._demo_plugin_mod")
        mod.TOOLS = [demo_tool]
        sys.modules["tests._demo_plugin_mod"] = mod
        try:
            assert pl.load_plugins() == 1
            from junjun_skills.registry import get_tools
            assert any(t.name == "demo_tool" for t in get_tools())
        finally:
            del sys.modules["tests._demo_plugin_mod"]

    def test_probe_failure_disables(self, tmp_path, monkeypatch):
        import junjun_skills.plugin_loader as pl
        plug = tmp_path / "broken"
        plug.mkdir()
        (plug / "_manifest.json").write_text(json.dumps({
            "name": "broken", "module": "tests._broken_plugin", "tools_attr": "TOOLS",
        }), encoding="utf-8")
        monkeypatch.setattr(pl, "PLUGINS_DIR", tmp_path)

        import sys, types
        mod = types.ModuleType("tests._broken_plugin")
        mod.TOOLS = []
        mod.probe_available = lambda: False
        sys.modules["tests._broken_plugin"] = mod
        try:
            assert pl.load_plugins() == 0
        finally:
            del sys.modules["tests._broken_plugin"]

    def test_whitelist_gates_session(self, tmp_path, monkeypatch):
        import junjun_skills.plugin_loader as pl
        plug = tmp_path / "gated"
        plug.mkdir()
        (plug / "_manifest.json").write_text(json.dumps({
            "name": "gated", "module": "tests._gated_plugin", "tools_attr": "TOOLS",
            "available_for": ["qq:888:group"],
        }), encoding="utf-8")
        monkeypatch.setattr(pl, "PLUGINS_DIR", tmp_path)

        import sys, types
        from langchain_core.tools import tool

        @tool
        def gated_tool() -> str:
            """gated。"""
            return "ok"
        mod = types.ModuleType("tests._gated_plugin")
        mod.TOOLS = [gated_tool]
        sys.modules["tests._gated_plugin"] = mod
        try:
            pl.load_plugins()
            from junjun_skills.registry import get_tools

            class S:
                chat_id = "qq:888:group"

            class Other:
                chat_id = "qq:999:group"
            assert any(t.name == "gated_tool" for t in get_tools(S()))
            assert not any(t.name == "gated_tool" for t in get_tools(Other()))
        finally:
            del sys.modules["tests._gated_plugin"]

    def test_bad_manifest_skipped(self, tmp_path, monkeypatch):
        import junjun_skills.plugin_loader as pl
        plug = tmp_path / "bad"
        plug.mkdir()
        (plug / "_manifest.json").write_text("not json{", encoding="utf-8")
        monkeypatch.setattr(pl, "PLUGINS_DIR", tmp_path)
        assert pl.load_plugins() == 0  # 不崩


class TestRelationshipServerTools:
    """直接调 server 的工具函数（存储与画像同库验证）。"""

    def test_penalty_writes_profile(self):
        from junjun_mcp_server.relationship_mcp_server import apply_relationship_penalty
        out = apply_relationship_penalty("111", "qq", "insult", "severe", "骂人")
        assert "-15" in out  # -10 * 1.5
        from junjun_memory.user_profile import get_profile_store
        points = get_profile_store().get_points("qq", "111")
        assert any("惩罚" in p["content"] for p in points)

    def test_impression_and_profile_roundtrip(self):
        from junjun_mcp_server.relationship_mcp_server import (
            update_user_impression, get_user_profile, set_user_name,
        )
        update_user_impression("222", "qq", "热心肠爱开玩笑", 0.9)
        set_user_name("222", "qq", "老王")
        profile = get_user_profile("222", "qq")
        assert "热心肠" in profile

    def test_unknown_penalty_type(self):
        from junjun_mcp_server.relationship_mcp_server import apply_relationship_penalty
        assert "未知" in apply_relationship_penalty("111", "qq", "nonsense")

    def test_empty_profile(self):
        from junjun_mcp_server.relationship_mcp_server import get_user_profile
        assert "暂无" in get_user_profile("nobody", "qq")
