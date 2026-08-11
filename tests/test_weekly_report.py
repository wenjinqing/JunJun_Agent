"""每周群报测试：周统计 / 成稿挂审 / 审批放行与驳回 / 超时默认不发 / 调度去重。

Messages 用 tmp 独立库（bind_ctx + create_tables），不触生产 junjun.db；
LLM/通知/发送全部打桩。
"""

import asyncio
import time
from datetime import datetime

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database.models import Messages
from junjun_skills.plugins.weekly_report import tools as wr

test_db = SqliteDatabase(":memory:")
CHAT = "qq:1158561385:group"
NOW = time.time()


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


def _msg(uid, nick, text, *, ts=None, emoji=False, pic=False, bot=False):
    Messages.create(bot_id="12345", message_id=f"m{ts}{uid}{text[:2]}",
                    chat_id=CHAT, time=ts or NOW, user_id=uid,
                    user_nickname=nick, processed_plain_text=text,
                    is_bot=bot, is_emoji=emoji, is_picid=pic)


def _night_ts(day_offset=0):
    """今天凌晨 3 点的时间戳（减 day_offset 天）。"""
    d = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
    return d.timestamp() - day_offset * 86400


@pytest.fixture
def env(tmp_path, monkeypatch):
    old = cfg_mod.global_config
    _set_config({"weekly_report": {"enable": True, "day": "sun", "time": "20:00",
                                   "min_messages": 5,
                                   "approval_timeout_seconds": 600},
                 "chat": {"group_list": [1158561385]},
                 "personality": {"personality": "你是君君", "reply_style": "口语"}})
    notices, sent = [], []
    import junjun_core.security as sec
    async def fake_notify(text):
        notices.append(text)
        return True
    monkeypatch.setattr(sec, "notify_admin", fake_notify)
    import junjun_agent.outbound as outbound
    async def fake_send(chat_id, segments, **kw):
        sent.append((chat_id, segments[0].data, kw))
    monkeypatch.setattr(outbound, "send_proactive", fake_send)
    wr._pending.clear()
    monkeypatch.setattr(wr, "DATA_DIR", tmp_path)
    with test_db.bind_ctx([Messages]):
        test_db.create_tables([Messages])
        Messages.delete().execute()
        yield type("E", (), {"notices": notices, "sent": sent})()
    for info in wr._pending.values():
        t = info.get("timeout_task")
        if t:
            t.cancel()
    wr._pending.clear()
    cfg_mod.global_config = old


class _FakeModel:
    def __init__(self, content="周报正文：名场面+颁奖词"):
        self._content = content

    async def ainvoke(self, messages, config=None):
        return type("R", (), {"content": self._content})()


def _seed_week():
    """一周素材：甲话痨+夜猫，乙表情帝，丙大文豪。"""
    for i in range(4):
        _msg("u1", "甲", f"话痨发言{i}")
    _msg("u1", "甲", "凌晨碎碎念", ts=_night_ts())
    for i in range(2):
        _msg("u2", "乙", f"图{i}", emoji=True)
    _msg("u3", "丙", "长" * 200)
    _msg("u1", "甲", "上周的旧消息", ts=NOW - 8 * 86400)  # 超窗，不计入


class TestWeekStats:
    def test_stats_and_awards(self, env):
        _seed_week()
        stats = wr._week_stats(CHAT, NOW - 7 * 86400)
        assert stats["total"] == 8, "超窗消息不计入"
        awards = {a[0]: a[1] for a in stats["awards"]}
        assert awards["话痨王"] == "甲"
        assert awards["夜猫子"] == "甲"
        assert awards["表情帝"] == "乙"
        assert awards["大文豪"] == "丙"

    def test_bot_messages_excluded(self, env):
        _seed_week()
        for i in range(20):
            _msg("12345", "君君", f"bot 刷屏{i}", bot=True)
        stats = wr._week_stats(CHAT, NOW - 7 * 86400)
        assert stats["total"] == 8, "bot 自己的消息不进统计"

    def test_single_user_no_awards(self, env):
        """只有一个人说话的群不发奖（自己给自己颁奖很尴尬）。"""
        _msg("u1", "甲", "自言自语1")
        _msg("u1", "甲", "自言自语2")
        stats = wr._week_stats(CHAT, NOW - 7 * 86400)
        assert stats["total"] == 2 and stats["awards"] == []


class TestRunFlow:
    @pytest.mark.asyncio
    async def test_quiet_week_skips(self, env):
        _msg("u1", "甲", "就一条")
        out = await wr.run_for_chat(CHAT, model=_FakeModel())
        assert "冷清" in out
        assert not wr._pending, "冷场周不烧模型不挂审批"

    @pytest.mark.asyncio
    async def test_happy_parks_then_approve_sends(self, env):
        _seed_week()
        out = await wr.run_for_chat(CHAT, model=_FakeModel())
        assert "审批" in out
        assert len(wr._pending) == 1
        assert env.notices and "周报正文" in env.notices[0]
        assert not env.sent, "审批前不许发群"
        key = next(iter(wr._pending))
        await wr.approve(key, True)
        assert env.sent and env.sent[0][0] == CHAT
        assert "周报正文" in env.sent[0][1]
        assert not wr._pending

    @pytest.mark.asyncio
    async def test_reject_not_sent(self, env):
        _seed_week()
        await wr.run_for_chat(CHAT, model=_FakeModel())
        key = next(iter(wr._pending))
        await wr.approve(key, False)
        assert not env.sent and not wr._pending

    @pytest.mark.asyncio
    async def test_timeout_defaults_no_send(self, env, monkeypatch):
        monkeypatch.setattr(wr, "_cfg",
                            lambda: {"approval_timeout_seconds": 0.05,
                                     "min_messages": 5})
        _seed_week()
        await wr.run_for_chat(CHAT, model=_FakeModel())
        await asyncio.sleep(0.15)
        assert not wr._pending and not env.sent, "超时默认不发"


class TestApprovalHook:
    def _meta(self, uid, text):
        return type("M", (), {"user_id": uid, "text": text})()

    def _session(self):
        return type("S", (), {"chat_id": "qq:99:private"})()

    @pytest.mark.asyncio
    async def test_gating(self, env, monkeypatch):
        import junjun_core.security as sec
        monkeypatch.setattr(sec, "is_admin", lambda uid: uid == "1")
        wr._pending["k1"] = {"chat_id": CHAT, "text": "正文"}
        # 非管理员 / 模糊词 / 空 pending 都不消费（误判回归）
        assert await wr.approval_hook(self._session(), self._meta("2", "发")) is False
        assert await wr.approval_hook(self._session(), self._meta("1", "发一下")) is False
        assert "k1" in wr._pending
        assert await wr.approval_hook(self._session(), self._meta("1", "发")) is True
        await asyncio.sleep(0)
        assert env.sent, "放行后正文发群"
        assert any("发到群里" in s[1] for s in env.sent), "要有 ack 回执"


class TestTick:
    @pytest.mark.asyncio
    async def test_fires_once_per_week(self, env, monkeypatch):
        from datetime import datetime
        day = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][datetime.now().weekday()]
        monkeypatch.setattr(wr, "_cfg",
                            lambda: {"enable": True, "day": day,
                                     "time": datetime.now().strftime("%H:%M"),
                                     "min_messages": 5})
        runs = []
        async def fake_run(chat_id, **kw):
            runs.append(chat_id)
        monkeypatch.setattr(wr, "run_for_chat", fake_run)
        await wr.weekly_report_tick()
        await wr.weekly_report_tick()
        assert runs == [CHAT], "同一周只跑一次"

    @pytest.mark.asyncio
    async def test_disabled_no_fire(self, env, monkeypatch):
        monkeypatch.setattr(wr, "_cfg", lambda: {"enable": False})
        runs = []
        monkeypatch.setattr(wr, "run_for_chat", lambda *a, **kw: runs.append(a))
        await wr.weekly_report_tick()
        assert not runs
