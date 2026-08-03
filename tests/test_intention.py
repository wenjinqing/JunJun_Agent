"""意向系统（P7）：spawn 去重/淘汰/过期、评估门全规则、执行链路、熔断、三类生成源。

验收场景（doc）：「昨晚 A 说考试焦虑」-> 今晨队列出现「想问问 A 考得怎样」并发出。
"""

import time
from datetime import datetime

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m
from junjun_agent.loop import intention as itm


@pytest.fixture
def env(monkeypatch, tmp_path):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"intention": {"enable": True, "quiet_hours": "23:00-08:00",
                           "max_pending_per_chat": 2, "max_per_chat_per_day": 2,
                           "max_global_per_day": 10, "min_age_minutes": 60,
                           "min_intimacy": 30, "morning_time": "09:00"}})
    monkeypatch.setattr(itm, "_STATE_PATH", tmp_path / "intention_state.json")
    monkeypatch.setattr(itm, "_in_quiet_hours", lambda: False)  # 测试时间无关
    monkeypatch.setattr(itm.random, "random", lambda: 0.9)      # 抖动放行
    # 亲密度默认熟人（care 门槛放行）；低分拦截用例自行覆盖
    import junjun_express.intimacy as intim
    monkeypatch.setattr(intim, "get_intimacy", lambda uid: (80.0, 10, "熟人"))
    db = SqliteDatabase(":memory:")
    with db.bind_ctx([m.Intention, m.Messages]):
        db.create_tables([m.Intention, m.Messages])
        yield monkeypatch
    cfg_mod.global_config = old


def _spawn(kind="care_followup", chat_id="qq:1:group", motive="想关心一下",
           user_id="111", priority=3, ttl=12):
    return itm.spawn(kind, chat_id, motive, user_id=user_id,
                     user_nickname="甲", priority=priority, ttl_hours=ttl)


def _age(hours=4):
    """把全部 pending 意向的创建时间往回拨（过 min_age 门槛）。"""
    for it in m.Intention.select():
        it.created_at = time.time() - hours * 3600
        it.save()


class _GenModel:
    async def ainvoke(self, msgs, config=None):
        return type("R", (), {"content": "甲，后来好点了吗？"})()


class _JudgeOk:
    async def ainvoke(self, msgs, config=None):
        return type("R", (), {"content": "合适"})()


class _JudgeNo:
    async def ainvoke(self, msgs, config=None):
        return type("R", (), {"content": "不合适"})()


def _fake_gw(monkeypatch, fail=False):
    sent = []

    class _GW:
        async def send_reply(self, rs):
            if fail:
                raise RuntimeError("napcat down")
            sent.append(rs)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _GW())
    return sent


class TestSpawn:
    def test_spawn_and_dedupe(self, env):
        assert _spawn() is True
        assert _spawn(motive="新动机") is False  # 去重：同类同会话同人
        it = m.Intention.get()
        assert it.motive == "新动机" and it.status == "pending"

    def test_cap_priority_evict(self, env):
        """满员（2）时：更高优先级挤掉最差的；更低优先级直接弃。"""
        _spawn(kind="a", priority=5)
        _spawn(kind="b", priority=6)
        assert _spawn(kind="c", priority=2) is True   # 挤掉 b(6)
        kinds = {it.kind: it.status for it in m.Intention.select()}
        assert kinds["b"] == "dropped"
        assert _spawn(kind="d", priority=9) is False  # 不如在队的，弃

    def test_disabled_no_spawn(self, env):
        cfg_mod.global_config.raw["intention"]["enable"] = False
        assert _spawn() is False

    def test_expire_sweep(self, env):
        _spawn(ttl=0)  # 立刻过期
        assert itm.expire_sweep() == 1
        assert m.Intention.get().status == "expired"


class TestCareHook:
    def test_emo_spawns_care(self, env):
        assert itm.spawn_care_if_needed("qq:1:group", "111", "甲", "考砸了，好难受") is True
        it = m.Intention.get()
        assert it.kind == "care_followup" and "甲" in it.motive and "考砸" in it.motive

    def test_normal_text_no_spawn(self, env):
        assert itm.spawn_care_if_needed("qq:1:group", "111", "甲", "今晚吃什么") is False
        assert itm.spawn_care_if_needed("qq:1:group", "111", "甲", "") is False

    def test_disabled_hook_noop(self, env):
        cfg_mod.global_config.raw["intention"]["enable"] = False
        assert itm.spawn_care_if_needed("qq:1:group", "111", "甲", "考砸了") is False


class TestEvaluateGate:
    def test_too_fresh_blocked(self, env):
        _spawn()
        it = m.Intention.get()
        ok, reason = itm.evaluate(it)
        assert not ok and reason == "too_fresh"

    def test_aged_care_passes(self, env):
        _spawn()
        _age(4)
        it = m.Intention.get()
        ok, reason = itm.evaluate(it)
        assert ok, reason

    def test_quiet_hours_blocked(self, env, monkeypatch):
        monkeypatch.setattr(itm, "_in_quiet_hours", lambda: True)
        _spawn()
        _age(4)
        ok, reason = itm.evaluate(m.Intention.get())
        assert not ok and reason == "quiet_hours"

    def test_daily_caps(self, env):
        _spawn()
        _age(4)
        it = m.Intention.get()
        # 本会话今天已发 2 条 -> 拦
        for i in range(2):
            m.Intention.create(kind="x", chat_id="qq:1:group", motive="m",
                               priority=5, status="fired", fired_at=time.time(),
                               expires_at=time.time() + 1)
        ok, reason = itm.evaluate(it)
        assert not ok and reason == "chat_daily_cap"

    def test_same_kind_dedupe(self, env):
        _spawn()
        _age(4)
        m.Intention.create(kind="care_followup", chat_id="qq:1:group", motive="m",
                           priority=5, status="fired", fired_at=time.time(),
                           expires_at=time.time() + 1)
        ok, reason = itm.evaluate(m.Intention.get())
        assert not ok and reason == "same_kind_fired_24h"

    def test_low_intimacy_blocked(self, env, monkeypatch):
        import junjun_express.intimacy as intim
        monkeypatch.setattr(intim, "get_intimacy", lambda uid: (10.0, 1, "陌生"))
        _spawn()
        _age(4)
        ok, reason = itm.evaluate(m.Intention.get())
        assert not ok and reason == "low_intimacy"


class TestFire:
    @pytest.mark.asyncio
    async def test_fire_sends_and_marks(self, env):
        sent = _fake_gw(env)
        _spawn()
        _age(4)
        it = m.Intention.get()
        assert await itm._fire(it, gen_model=_GenModel(), judge_model=_JudgeOk())
        it = m.Intention.get_by_id(it.id)
        assert it.status == "fired"
        assert sent and "好点了吗" in sent[0].segments[0].data

    @pytest.mark.asyncio
    async def test_judge_veto_drops(self, env):
        sent = _fake_gw(env)
        _spawn()
        _age(4)
        it = m.Intention.get()
        assert await itm._fire(it, gen_model=_GenModel(), judge_model=_JudgeNo()) is False
        assert m.Intention.get_by_id(it.id).status == "dropped"
        assert not sent

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, env):
        _fake_gw(env, fail=True)
        _spawn()
        _age(4)
        it = m.Intention.get()
        for _ in range(3):
            await itm._fire(it, gen_model=_GenModel(), judge_model=_JudgeOk())
        assert itm._circuit_open() is True
        ok, reason = itm.evaluate(m.Intention.get())
        assert not ok and reason == "circuit_open"


class TestE2EAcceptance:
    @pytest.mark.asyncio
    async def test_emo_last_night_to_care_message(self, env):
        """验收场景：昨晚 A 说考试焦虑 -> 队列有关心意向 -> 今晨过闸发出。"""
        sent = _fake_gw(env)
        itm.spawn_care_if_needed("qq:1:group", "111", "甲", "考试焦虑，睡不着")
        it = m.Intention.get()
        assert it.kind == "care_followup"
        _age(4)  # 今晨
        fired = await itm.intention_tick(gen_model=_GenModel(), judge_model=_JudgeOk())
        assert fired == 1
        assert "好点了吗" in sent[0].segments[0].data


class TestGenerators:
    @pytest.mark.asyncio
    async def test_on_diary_written(self, env):
        m.Messages.create(chat_id="qq:1:group", user_id="111", user_nickname="甲",
                          time=time.time(), message_id="x1",
                          processed_plain_text="聊天", bot_id="1")

        class _PlanModel:
            async def ainvoke(self, msgs, config=None):
                return type("R", (), {"content": '[{"motive": "问问甲考试怎样"}]'})()

        n = await itm.on_diary_written("今天甲好像考试焦虑……", model=_PlanModel())
        assert n == 1
        it = m.Intention.get()
        assert it.kind == "diary_plan" and "考试" in it.motive

    def test_morning_greet_dedupe_per_day(self, env, monkeypatch):
        class _FakeDT:
            @staticmethod
            def now():
                return datetime(2026, 8, 3, 9, 30)
        monkeypatch.setattr(itm, "datetime", _FakeDT)
        m.Messages.create(chat_id="qq:1:group", user_id="111", user_nickname="甲",
                          time=time.time(), message_id="x1",
                          processed_plain_text="聊天", bot_id="1")
        assert itm.spawn_scheduled_checks() == 1
        assert itm.spawn_scheduled_checks() == 0  # 今天排过了
