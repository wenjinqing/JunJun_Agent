"""阶段 7 单测：WebUI API（TestClient）+ DB 清理。"""

import time
import tomllib

import pytest
from peewee import SqliteDatabase
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_config_persist(monkeypatch, tmp_path):
    """persist_bot_config 默认写 config/bot_config.toml（生产配置！）——
    2026-08-13 实锤：test_hot_change_applies_to_memory 把测试值 66 写进了
    真实 max_context_size。隔离到 tmp 并放一个真 toml，写回路径照常走通。"""
    import junjun_core.config.config as cfg_mod
    cfg_file = tmp_path / "bot_config.toml"
    cfg_file.write_text("[plugins]\ndisabled = []\n", encoding="utf-8")
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", tmp_path)
    return cfg_file


@pytest.fixture(autouse=True)
def _memory_db(monkeypatch, tmp_path):
    """TestClient 在独立线程跑 app，:memory: 库跨线程不可见——用临时文件库。"""
    import junjun_core.database.models as m
    test_db = SqliteDatabase(str(tmp_path / "test.db"), pragmas={"journal_mode": "wal"})
    with test_db.bind_ctx(m.ALL_TABLES):
        test_db.create_tables(m.ALL_TABLES)
        monkeypatch.setattr(m, "db", test_db)
        import junjun_core.database as pkg
        monkeypatch.setattr(pkg, "db", test_db)
        yield test_db
    test_db.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("WEBUI_TOKEN", raising=False)
    from junjun_webui.server import app
    return TestClient(app)  # TestClient 来源 host 是 testclient，走 token 分支需显式设


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("WEBUI_TOKEN", "secret123")
    from junjun_webui.server import app
    c = TestClient(app)
    c.headers["Authorization"] = "Bearer secret123"
    return c


class TestAuth:
    def test_no_token_rejects_remote(self, client):
        # TestClient 默认 host=testclient（非 127.0.0.1）-> 403
        assert client.get("/api/stats").status_code == 403

    def test_wrong_token_401(self, monkeypatch):
        monkeypatch.setenv("WEBUI_TOKEN", "secret123")
        from junjun_webui.server import app
        c = TestClient(app)
        c.headers["Authorization"] = "Bearer wrong"
        assert c.get("/api/stats").status_code == 401

    def test_valid_token_ok(self, auth_client):
        assert auth_client.get("/api/stats").status_code == 200


class TestConfigAPI:
    def test_get_returns_whitelist_keys(self, auth_client):
        cfg = auth_client.get("/api/config").json()
        assert "chat" in cfg and "max_context_size" in cfg["chat"]

    def test_hot_change_applies_to_memory(self, auth_client, _fake_bot_config):
        r = auth_client.post("/api/config", json={"chat": {"max_context_size": 66}})
        assert r.status_code == 200
        assert _fake_bot_config.raw["chat"]["max_context_size"] == 66

    def test_non_whitelist_key_rejected(self, auth_client):
        r = auth_client.post("/api/config", json={"gateway": {"port": 1}})
        assert r.status_code == 400


class TestPluginTogglePersist:
    """2026-08-13 审查 P2：WebUI 插件开关此前只改内存，重启即丢——
    插件级开关默认写回 bot_config.toml [plugins].disabled。"""

    @staticmethod
    def _register_fake_plugin():
        from langchain_core.tools import tool
        from junjun_skills import registry

        @tool
        def fake_plug_tool(x: str) -> str:
            """测试插件工具。

            Args:
                x: 输入
            """
            return x

        registry.register(fake_plug_tool, plugin="fakeplug")

    def test_disable_persists_to_toml(self, auth_client, _fake_bot_config,
                                      _isolate_config_persist):
        from junjun_skills import registry
        self._register_fake_plugin()
        r = auth_client.post("/api/plugins/fakeplug", json={"enabled": False})
        assert r.status_code == 200 and r.json()["persisted"] is True
        assert not registry.is_plugin_enabled("fakeplug")
        doc = tomllib.loads(_isolate_config_persist.read_text(encoding="utf-8"))
        assert doc["plugins"]["disabled"] == ["fakeplug"]

    def test_enable_removes_from_toml(self, auth_client, _fake_bot_config,
                                      _isolate_config_persist):
        from junjun_skills import registry
        self._register_fake_plugin()
        auth_client.post("/api/plugins/fakeplug", json={"enabled": False})
        r = auth_client.post("/api/plugins/fakeplug", json={"enabled": True})
        assert r.json()["persisted"] is True
        assert registry.is_plugin_enabled("fakeplug")
        doc = tomllib.loads(_isolate_config_persist.read_text(encoding="utf-8"))
        assert doc["plugins"]["disabled"] == []

    def test_skill_toggle_not_persisted(self, auth_client, _fake_bot_config,
                                        _isolate_config_persist):
        """skill 级开关无配置归宿，只改运行时——不许乱写 [plugins].disabled。"""
        self._register_fake_plugin()
        r = auth_client.post("/api/plugins/fake_plug_tool", json={"enabled": False})
        assert r.status_code == 200 and r.json()["persisted"] is False
        doc = tomllib.loads(_isolate_config_persist.read_text(encoding="utf-8"))
        assert doc["plugins"]["disabled"] == []

    def test_persist_false_skips_write(self, auth_client, _fake_bot_config,
                                       _isolate_config_persist):
        from junjun_skills import registry
        self._register_fake_plugin()
        r = auth_client.post("/api/plugins/fakeplug",
                             json={"enabled": False, "persist": False})
        assert r.json()["persisted"] is False
        assert not registry.is_plugin_enabled("fakeplug")   # 运行时生效
        doc = tomllib.loads(_isolate_config_persist.read_text(encoding="utf-8"))
        assert doc["plugins"]["disabled"] == []             # 文件不动

    def test_existing_disabled_union_kept(self, auth_client, _fake_bot_config,
                                          _isolate_config_persist):
        """配置里原就禁用的插件（从未加载、不在运行时集）不能被一次 toggle 抹掉。"""
        _fake_bot_config.raw["plugins"] = {"disabled": ["otherplug"]}
        self._register_fake_plugin()
        auth_client.post("/api/plugins/fakeplug", json={"enabled": False})
        doc = tomllib.loads(_isolate_config_persist.read_text(encoding="utf-8"))
        assert doc["plugins"]["disabled"] == ["fakeplug", "otherplug"]


class TestStatsSessions:
    def test_stats_counts(self, auth_client):
        from junjun_core.database import Messages, LLMUsage
        Messages.create(message_id="1", chat_id="c1", time=time.time(), is_bot=False)
        Messages.create(message_id="", chat_id="c1", time=time.time(), is_bot=True)
        LLMUsage.create(time=time.time(), request_type="agent",
                        prompt_tokens=100, completion_tokens=50)
        s = auth_client.get("/api/stats").json()
        assert s["received"] == 1 and s["replied"] == 1
        assert s["usage"][0]["prompt_tokens"] == 100

    def test_sessions_list(self, auth_client):
        from junjun_core.gateway.session_manager import get_session_manager, ChatSession
        get_session_manager()._sessions["qq:t:group"] = ChatSession("qq:t:group", "qq", group_id="t")
        rows = auth_client.get("/api/sessions").json()
        assert any(r["chat_id"] == "qq:t:group" for r in rows)


class TestDataManagement:
    def test_jargon_list_and_delete(self, auth_client):
        from junjun_core.database import Jargon
        row = Jargon.create(term="xswl", explanation="笑死我了", count=3)
        rows = auth_client.get("/api/jargon").json()
        assert rows[0]["term"] == "xswl"
        auth_client.delete(f"/api/jargon/{row.id}")
        assert auth_client.get("/api/jargon").json() == []

    def test_persons_with_points(self, auth_client):
        from junjun_memory.user_profile import get_profile_store
        get_profile_store().add_point("qq", "111", "喜好", "火锅", 0.9, nickname="甲")
        rows = auth_client.get("/api/persons").json()
        assert rows and "火锅" in "".join(rows[0]["points"])


class TestDBCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_only_old_llm_usage(self, _fake_bot_config):
        _fake_bot_config.raw["database"] = {
            "enable_auto_cleanup": True, "cleanup_retention_days": 60,
        }
        from junjun_core.database import LLMUsage, Messages
        from junjun_core.database.cleanup import run_cleanup
        LLMUsage.create(time=time.time() - 100 * 86400, request_type="old")
        LLMUsage.create(time=time.time(), request_type="new")
        Messages.create(message_id="1", chat_id="c1", time=time.time() - 100 * 86400)
        await run_cleanup()
        assert LLMUsage.select().count() == 1
        assert LLMUsage.select().first().request_type == "new"
        assert Messages.select().count() == 1  # messages 永不清

    @pytest.mark.asyncio
    async def test_cleanup_disabled(self, _fake_bot_config):
        _fake_bot_config.raw["database"] = {"enable_auto_cleanup": False}
        from junjun_core.database import LLMUsage
        from junjun_core.database.cleanup import run_cleanup
        LLMUsage.create(time=0, request_type="ancient")
        await run_cleanup()
        assert LLMUsage.select().count() == 1
