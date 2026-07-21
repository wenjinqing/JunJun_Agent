"""topic_finder 插件测试：定时/静默触发、生成发送、去重、降级、命令。"""

import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text):
    return SimpleNamespace(text=text, user_id="12345", nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


def _cfg_dict(**over):
    base = {
        "enable": True,
        "daily_times": ["12:30"],
        "target_groups": ["111", "222"],
        "min_interval_hours": 3,
        "silence_minutes": 60,
        "rss_feeds": ["https://x/rss"],
        "web_llm_enable": False,
    }
    base.update(over)
    return base


@pytest.fixture
def tf(monkeypatch, tmp_path, _fake_bot_config):
    """导入插件模块，隔离 DATA_DIR，并注入 [topic_finder] 配置。"""
    import junjun_skills.plugins.topic_finder.tools as mod
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    _fake_bot_config.raw["topic_finder"] = _cfg_dict()
    return mod


@pytest.fixture
def mock_sources(monkeypatch, tf):
    """隔离 LLM/RSS/HTTP：素材固定两条，生成固定话题。"""
    async def _rss(cfg):
        return ["标题A", "标题B"]

    async def _web(cfg):
        return []

    async def _gen(cfg, materials, recent=None):
        return "你们看到那个新闻了吗，有点离谱"

    monkeypatch.setattr(tf, "fetch_rss", _rss)
    monkeypatch.setattr(tf, "fetch_web_hot", _web)
    monkeypatch.setattr(tf, "generate_topic", _gen)
    return tf


def _freeze_now(monkeypatch, tf, hour, minute=0):
    """冻结模块内 datetime.now() 到指定时刻（日期固定，测试只关心时分）。"""
    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 21, hour, minute)

    monkeypatch.setattr(tf, "datetime", _FakeDT)


def _bind_messages(tmp_path):
    """peewee bind_ctx 临时 Messages 表（参照 test_action_skills 模式）。"""
    import peewee
    from junjun_core.database import models as m
    db = peewee.SqliteDatabase(str(tmp_path / "msg.db"))
    return db, m.Messages


@pytest.fixture
def _recent_msgs(tmp_path):
    """绑定临时 Messages 表并写入两群近期消息（隔离静默路径，不碰真实库）。"""
    db, Messages = _bind_messages(tmp_path)
    with db.bind_ctx([Messages]):
        db.create_tables([Messages])
        for gid in ("111", "222"):
            Messages.create(chat_id=f"qq:{gid}:group", time=time.time() - 60,
                            message_id=f"m-{gid}")
        yield


class TestDailySchedule:
    async def test_daily_hit_sends_all_groups(self, tf, mock_sources, _fake_gateway,
                                              monkeypatch, _recent_msgs):
        _freeze_now(monkeypatch, tf, 12, 30)
        await tf.topic_finder_tick()
        assert {r.target_group_id for r in _fake_gateway} == {"111", "222"}
        assert all(r.segments[0].type == "text" for r in _fake_gateway)
        # 同一时刻再 tick：当日该 slot 已记录，不重复发
        await tf.topic_finder_tick()
        assert len(_fake_gateway) == 2

    async def test_daily_miss_no_send(self, tf, mock_sources, _fake_gateway,
                                      monkeypatch, _recent_msgs):
        _freeze_now(monkeypatch, tf, 13, 0)
        await tf.topic_finder_tick()
        assert _fake_gateway == []


class TestSilenceTrigger:
    async def test_silent_group_triggers(self, tf, mock_sources, _fake_gateway,
                                         monkeypatch, tmp_path, _fake_bot_config):
        _fake_bot_config.raw["topic_finder"] = _cfg_dict(daily_times=[])
        _freeze_now(monkeypatch, tf, 15, 0)
        db, Messages = _bind_messages(tmp_path)
        with db.bind_ctx([Messages]):
            db.create_tables([Messages])
            # 群 111 两小时前有人说话（超过 silence_minutes=60），群 222 无记录 -> 都静默
            Messages.create(chat_id="qq:111:group", time=time.time() - 7200, message_id="m1")
            await tf.topic_finder_tick()
        assert {r.target_group_id for r in _fake_gateway} == {"111", "222"}

    async def test_active_group_not_triggered(self, tf, mock_sources, _fake_gateway,
                                              monkeypatch, tmp_path, _fake_bot_config):
        _fake_bot_config.raw["topic_finder"] = _cfg_dict(daily_times=[])
        _freeze_now(monkeypatch, tf, 15, 0)
        db, Messages = _bind_messages(tmp_path)
        with db.bind_ctx([Messages]):
            db.create_tables([Messages])
            # 两个群都有近期消息 -> 不触发
            for gid in ("111", "222"):
                Messages.create(chat_id=f"qq:{gid}:group", time=time.time() - 60,
                                message_id=f"m-{gid}")
            await tf.topic_finder_tick()
        assert _fake_gateway == []

    async def test_min_interval_blocks(self, tf, mock_sources, _fake_gateway,
                                       monkeypatch, tmp_path, _fake_bot_config):
        _fake_bot_config.raw["topic_finder"] = _cfg_dict(daily_times=[])
        _freeze_now(monkeypatch, tf, 15, 0)
        # 本插件刚刚在 111 发过言（不足 min_interval_hours=3）
        tf._write_json("last_send.json", {"groups": {"111": time.time()}})
        db, Messages = _bind_messages(tmp_path)
        with db.bind_ctx([Messages]):
            db.create_tables([Messages])
            Messages.create(chat_id="qq:111:group", time=time.time() - 7200, message_id="m1")
            await tf.topic_finder_tick()
        # 111 被最小间隔拦下；222 无记录视为静默，正常触发
        assert {r.target_group_id for r in _fake_gateway} == {"222"}

    async def test_quiet_hours_no_silence_trigger(self, tf, mock_sources, _fake_gateway,
                                                  monkeypatch, tmp_path, _fake_bot_config):
        _fake_bot_config.raw["topic_finder"] = _cfg_dict(daily_times=[])
        _freeze_now(monkeypatch, tf, 3, 0)  # 02:00-06:00 深夜不打扰
        db, Messages = _bind_messages(tmp_path)
        with db.bind_ctx([Messages]):
            db.create_tables([Messages])
            Messages.create(chat_id="qq:111:group", time=time.time() - 7200, message_id="m1")
            await tf.topic_finder_tick()
        assert _fake_gateway == []


class TestDedupAndDegrade:
    async def test_duplicate_topic_skipped(self, tf, mock_sources, _fake_gateway,
                                           monkeypatch, _recent_msgs):
        _freeze_now(monkeypatch, tf, 12, 30)
        # 生成结果与最近话题重复（重试仍重复）-> 本轮跳过
        tf._write_json("recent_topics.json",
                       [{"content": "你们看到那个新闻了吗，有点离谱", "ts": time.time() - 3600}])
        await tf.topic_finder_tick()
        assert _fake_gateway == []

    async def test_no_materials_skip(self, tf, _fake_gateway, monkeypatch, _recent_msgs):
        async def _none(cfg):
            return []

        monkeypatch.setattr(tf, "fetch_rss", _none)
        monkeypatch.setattr(tf, "fetch_web_hot", _none)
        _freeze_now(monkeypatch, tf, 12, 30)
        await tf.topic_finder_tick()
        assert _fake_gateway == []

    async def test_llm_failure_falls_back_to_title(self, tf, _fake_gateway,
                                                   monkeypatch, _recent_msgs):
        async def _rss(cfg):
            return ["标题A"]

        async def _web(cfg):
            return []

        async def _gen_fail(cfg, materials, recent=None):
            return None

        monkeypatch.setattr(tf, "fetch_rss", _rss)
        monkeypatch.setattr(tf, "fetch_web_hot", _web)
        monkeypatch.setattr(tf, "generate_topic", _gen_fail)
        _freeze_now(monkeypatch, tf, 12, 30)
        await tf.topic_finder_tick()
        assert len(_fake_gateway) == 2
        assert "标题A" in _fake_gateway[0].segments[0].data

    async def test_enable_false_no_action(self, tf, mock_sources, _fake_gateway,
                                          monkeypatch, _fake_bot_config, _recent_msgs):
        _fake_bot_config.raw["topic_finder"] = _cfg_dict(enable=False)
        _freeze_now(monkeypatch, tf, 12, 30)
        await tf.topic_finder_tick()
        assert _fake_gateway == []


class TestTopicTestCommand:
    async def test_command_registered(self, monkeypatch, tmp_path, _fake_bot_config):
        _fake_bot_config.raw["topic_finder"] = _cfg_dict()
        import importlib
        import junjun_skills.plugins.topic_finder.tools as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        importlib.reload(mod)  # reload 重新执行装饰器注册
        cmds = {c["name"]: c for c in commands.list_commands()}
        assert "topic_test" in cmds and cmds["topic_test"]["plugin"] == "topic_finder"

    async def test_topic_test_sends_to_current_chat(self, tf, mock_sources, _fake_gateway):
        result = await tf.topic_test_cmd(_ctx("/topic_test"))
        assert result is None  # 已通过 ctx.send 发送
        assert len(_fake_gateway) == 1
        assert _fake_gateway[0].target_group_id == "999"
        assert _fake_gateway[0].segments[0].type == "text"

    async def test_topic_test_no_materials(self, tf, monkeypatch):
        async def _none(cfg):
            return []

        monkeypatch.setattr(tf, "fetch_rss", _none)
        monkeypatch.setattr(tf, "fetch_web_hot", _none)
        result = await tf.topic_test_cmd(_ctx("/topic_test"))
        assert "素材" in result
