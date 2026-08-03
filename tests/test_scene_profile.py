"""跨场景用户档案（P6-4）：蒸馏打标/场合过滤/多群隔离/遗忘集成。

对抗验收（隐私生命线）：群聊注入块绝不出现私聊来源内容；
A 群的事不在 B 群说；「私聊很熟」只传达熟度不传达内容。
"""

import time

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m
from junjun_memory import scene_profile as sp


@pytest.fixture
def env(monkeypatch, tmp_path):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"scene_profile": {"enable": True, "min_messages": 3, "max_facts": 2}})
    monkeypatch.setattr(sp, "_STATE_PATH", tmp_path / "sp_state.json")
    db = SqliteDatabase(":memory:")
    with db.bind_ctx([m.Messages, m.UserSceneProfile]):
        db.create_tables([m.Messages, m.UserSceneProfile])
        yield monkeypatch
    cfg_mod.global_config = old


def _msg(user_id, chat_id, text, ts=None):
    m.Messages.create(chat_id=chat_id, user_id=user_id, user_nickname="甲",
                      time=ts or time.time(), message_id=f"m{text[:4]}{time.time_ns() % 9999}",
                      processed_plain_text=text, bot_id="10000001")


def _fact(user_id, content, scene, chat_id, updated_at=None):
    now = time.time()
    m.UserSceneProfile.create(person_id=f"qq:{user_id}", platform="qq",
                              user_id=user_id, content=content,
                              source_scene=scene, source_chat_id=chat_id,
                              weight=1.0, created_at=now,
                              updated_at=updated_at or now)


class _SceneAwareModel:
    """按 prompt 里的场景描述返回不同事实（模拟分场景蒸馏）。"""

    async def ainvoke(self, msgs, config=None):
        prompt = str(msgs[-1].content)
        if "私聊" in prompt:
            return type("R", (), {"content": '[{"fact": "他在准备考研"}]'})()
        return type("R", (), {"content": '[{"fact": "他打绝区零"}]'})()


class TestDistill:
    @pytest.mark.asyncio
    async def test_scene_tags_are_mechanical(self, env):
        """来源标签机械打标：私聊场景蒸出的事实一定带 private+私聊 chat_id。"""
        for i in range(3):
            _msg("111", "qq:1:group", f"绝区零真好玩{i}")
            _msg("111", "qq:111:private", f"考研复习到第{i}章")
        added = await sp.distill_user("qq", "111", model=_SceneAwareModel())
        assert added == 2
        rows = list(m.UserSceneProfile.select())
        by_scene = {r.source_scene: r for r in rows}
        assert by_scene["private"].content == "他在准备考研"
        assert by_scene["private"].source_chat_id == "qq:111:private"
        assert by_scene["group"].source_chat_id == "qq:1:group"

    @pytest.mark.asyncio
    async def test_too_few_lines_skipped(self, env):
        _msg("111", "qq:1:group", "就一句")
        assert await sp.distill_user("qq", "111", model=_SceneAwareModel()) == 0

    def test_parse_facts(self):
        assert sp._parse_facts('[{"fact": "a"}, {"fact": "b"}]') == ["a", "b"]
        assert sp._parse_facts("不会吧") == []
        assert sp._parse_facts('[{"no": "fact"}]') == []


class TestPrivacyBoundary:
    """对抗验收：群里套话也不能从注入块里拿到私聊/别群内容。"""

    def test_group_block_never_leaks_private(self, env):
        _fact("111", "他在准备考研", "private", "qq:111:private")
        _fact("111", "他打绝区零", "group", "qq:1:group")
        block = sp.build_scene_block("qq", "111", "qq:1:group", is_group=True)
        assert "他打绝区零" in block
        assert "考研" not in block           # 私聊事实绝不进群
        assert "私聊" not in block or "绝不外说" in block

    def test_multi_group_isolation(self, env):
        """A 群的事不在 B 群说。"""
        _fact("111", "他在A群聊过装修", "group", "qq:2:group")
        block = sp.build_scene_block("qq", "111", "qq:1:group", is_group=True)
        assert "装修" not in block

    def test_group_closeness_hint_without_content(self, env):
        """「私聊很熟」只传达熟度：提示出现但私聊事实内容不出现。"""
        _fact("111", "他私下在准备考试", "private", "qq:111:private")
        for i in range(25):
            _msg("111", "qq:111:private", f"私聊消息{i}")
        block = sp.build_scene_block("qq", "111", "qq:1:group", is_group=True)
        assert "你们私聊也很熟" in block
        assert "考试" not in block

    def test_private_block_sees_group_facts(self, env):
        """私聊里可以引用 ta 在群里的公开表现。"""
        _fact("111", "他打绝区零", "group", "qq:1:group")
        _fact("111", "他在准备考研", "private", "qq:111:private")
        block = sp.build_scene_block("qq", "111", "qq:111:private", is_group=False)
        assert "他打绝区零" in block and "他在准备考研" in block

    def test_disabled_returns_empty(self, env):
        cfg_mod.global_config.raw["scene_profile"]["enable"] = False
        _fact("111", "他打绝区零", "group", "qq:1:group")
        assert sp.build_scene_block("qq", "111", "qq:1:group", is_group=True) == ""


class TestForgetAndGc:
    def test_forget_scoped_to_current_chat(self, env):
        """非管理员 /忘掉：只删当前会话来源的档案事实。"""
        _fact("111", "他在本群聊过火锅", "group", "qq:1:group")
        _fact("111", "他私聊说过火锅", "private", "qq:111:private")
        n = sp.forget_user_facts("qq", "111", "火锅",
                                 admin=False, current_chat_id="qq:1:group")
        assert n == 1
        left = [r.content for r in m.UserSceneProfile.select()]
        assert left == ["他私聊说过火锅"]

    def test_forget_admin_all_scenes(self, env):
        _fact("111", "他在本群聊过火锅", "group", "qq:1:group")
        _fact("111", "他私聊说过火锅", "private", "qq:111:private")
        n = sp.forget_user_facts("qq", "111", "火锅", admin=True)
        assert n == 2

    def test_gc_stale(self, env):
        _fact("111", "新鲜事实", "group", "qq:1:group")
        _fact("111", "过期事实", "group", "qq:1:group",
              updated_at=time.time() - 15 * 86400)
        assert sp._gc_stale() == 1
        assert [r.content for r in m.UserSceneProfile.select()] == ["新鲜事实"]
        # 过期事实也不再注入
        _fact("111", "过期注入", "group", "qq:1:group",
              updated_at=time.time() - 15 * 86400)
        block = sp.build_scene_block("qq", "111", "qq:1:group", is_group=True)
        assert "过期注入" not in block


class TestTick:
    @pytest.mark.asyncio
    async def test_tick_gates(self, env):
        """活跃门槛 + 12h 节流：不达标/刚蒸过的不重复蒸。"""
        calls = []

        async def _distill(platform, user_id, **kw):
            calls.append(user_id)
            return 0
        env.setattr(sp, "distill_user", _distill)
        for i in range(3):  # min_messages=3
            _msg("111", "qq:1:group", f"话{i}")
        _msg("222", "qq:1:group", "只有一句")  # 不达标
        await sp.profile_tick()
        assert calls == ["111"]
        await sp.profile_tick()  # 12h 内不再蒸
        assert calls == ["111"]

    @pytest.mark.asyncio
    async def test_tick_disabled(self, env):
        cfg_mod.global_config.raw["scene_profile"]["enable"] = False
        calls = []

        async def _distill(platform, user_id, **kw):
            calls.append(user_id)
            return 0
        env.setattr(sp, "distill_user", _distill)
        for i in range(5):
            _msg("111", "qq:1:group", f"话{i}")
        await sp.profile_tick()
        assert calls == []
