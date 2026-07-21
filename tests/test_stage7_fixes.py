"""2026-07-21 验收修复批：timing gate / 速率限制 / 热改持久化 / 识图降级 / 在线时长 / 插件开关。"""

import asyncio
import time

import pytest


class TestRateLimit:
    def test_token_bucket(self, _fake_bot_config):
        from junjun_core.gateway import rate_limit
        rate_limit.reset()
        _fake_bot_config.raw["gateway"] = {"rate_limit_capacity": 2, "rate_limit_refill_per_sec": 0.0}
        assert rate_limit.allow_message("c1") is True
        assert rate_limit.allow_message("c1") is True
        assert rate_limit.allow_message("c1") is False  # 桶空
        assert rate_limit.allow_message("c2") is True   # 其他会话不受影响

    def test_refill(self, _fake_bot_config):
        from junjun_core.gateway import rate_limit
        rate_limit.reset()
        _fake_bot_config.raw["gateway"] = {"rate_limit_capacity": 1, "rate_limit_refill_per_sec": 100.0}
        assert rate_limit.allow_message("c1") is True
        time.sleep(0.05)  # 100/s -> 0.05s 补 5 个
        assert rate_limit.allow_message("c1") is True


class TestTimingGate:
    @pytest.mark.asyncio
    async def test_disabled_by_default(self, _fake_bot_config):
        from junjun_agent.funnel.session_queue import _timing_gate_wait
        assert _timing_gate_wait() == 0.0

    @pytest.mark.asyncio
    async def test_burst_merged_into_one(self, _fake_bot_config):
        from junjun_agent.funnel.session_queue import SessionQueue
        _fake_bot_config.raw["chat"]["enable_timing_gate"] = True
        _fake_bot_config.raw["chat"]["timing_gate_wait_seconds"] = 0.1

        handled = []

        class _Meta:
            def __init__(self, text):
                self.text = text

        class _Session:
            chat_id = "c1"

        async def handler(session, meta):
            handled.append(meta.text)

        q = SessionQueue("c1", handler)
        for i in range(3):
            q.put(_Session(), _Meta(f"msg{i}"))
        await asyncio.sleep(0.6)
        await q.stop()
        assert handled == ["msg2"]  # 连发聚拢，只评估最新一条


class TestConfigHotReload:
    def test_persist_changed_keys(self, tmp_path, _fake_bot_config):
        from junjun_core.config.config import persist_bot_config
        toml = tmp_path / "bot_config.toml"
        toml.write_text(
            '[chat]\ntalk_value = 0.9\nqq = "${MAIBOT_QQ_ACCOUNT}"\n', encoding="utf-8")
        _fake_bot_config.raw["chat"]["talk_value"] = 0.3
        _fake_bot_config.raw["chat"]["qq"] = "999"
        persist_bot_config([("chat", "talk_value"), ("chat", "qq")], path=toml)
        text = toml.read_text(encoding="utf-8")
        assert "talk_value = 0.3" in text
        assert "${MAIBOT_QQ_ACCOUNT}" in text  # 占位符键不被固化

    def test_listener_notified(self):
        from junjun_core.config.config import register_config_listener, notify_config_changed
        got = []
        register_config_listener(lambda changed: got.extend(changed))
        notify_config_changed(["chat.talk_value=0.3"])
        assert got == ["chat.talk_value=0.3"]


class TestVisionDegrade:
    @pytest.mark.asyncio
    async def test_no_vlm_degrades_to_placeholder(self, tmp_path, monkeypatch):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "v.db"))
        with db.bind_ctx([m.Images]):
            db.create_tables([m.Images])

            import junjun_memory.vision as vision
            monkeypatch.setattr(vision, "_get_vlm", lambda: None)  # VLM 未配置

            async def _fake_download(url):
                return b"img-bytes"

            monkeypatch.setattr(vision, "_download", _fake_download)
            out = await vision.describe_images(["http://x/1.png"])
            assert out == {"http://x/1.png": "[图片]"}
            assert vision.render_image_block(out) == ""

    @pytest.mark.asyncio
    async def test_cached_hash_skips_vlm(self, tmp_path, monkeypatch):
        import hashlib
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "v2.db"))
        with db.bind_ctx([m.Images]):
            db.create_tables([m.Images])
            h = hashlib.md5(b"img-bytes").hexdigest()
            m.Images.create(image_hash=h, description="一只猫", timestamp=1.0)

            import junjun_memory.vision as vision

            async def _fake_download(url):
                return b"img-bytes"

            monkeypatch.setattr(vision, "_download", _fake_download)
            out = await vision.describe_images(["http://x/1.png"], model=object())
            assert out["http://x/1.png"] == "一只猫"  # 命中缓存不调 VLM
            assert "一只猫" in vision.render_image_block(out)


class TestOnlineTimeRecord:
    @pytest.mark.asyncio
    async def test_record_extends(self, tmp_path, monkeypatch):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "o.db"))
        with db.bind_ctx([m.OnlineTime]):
            db.create_tables([m.OnlineTime])

            import junjun_agent.loop.statistics as stats
            stats._online_record_id = None
            await stats.record_online_time()
            first_end = m.OnlineTime.get().end_timestamp
            await stats.record_online_time()
            assert m.OnlineTime.select().count() == 1  # 续期而非新开
            assert m.OnlineTime.get().end_timestamp >= first_end


class TestPluginToggle:
    def test_disable_filters_tools(self):
        from junjun_skills import registry
        registry.load_builtin()
        assert any(t.name == "get_time" for t in registry.get_tools())
        assert registry.set_enabled("get_time", False) is True
        assert all(t.name != "get_time" for t in registry.get_tools())
        skills = registry.list_skills()
        entry = next(s for s in skills if s["name"] == "get_time")
        assert entry["enabled"] is False
        registry.set_enabled("get_time", True)
        assert registry.set_enabled("不存在的skill", True) is False
