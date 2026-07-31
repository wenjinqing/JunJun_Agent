"""subscription 插件测试：目标解析 / 基线 / 更新检测 / 推送 / 权限。

DB 用内存库隔离；pixiv/bilibili 抓取与 gateway 全部 monkeypatch。
"""

import time
from types import SimpleNamespace

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m

test_db = SqliteDatabase(":memory:")
TABLES = [m.Subscription]


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


@pytest.fixture
def _env(monkeypatch):
    old = cfg_mod.global_config
    _set_config({"subscription": {"enable": True}})
    with test_db.bind_ctx(TABLES):
        test_db.create_tables(TABLES)
        m.Subscription.delete().execute()
        import junjun_skills.plugins.subscription.tools as sub
        yield sub
    cfg_mod.global_config = old


def _mk_sub(sub, **kw):
    defaults = dict(kind="pixiv_author", target_id="16689973", target_name="某作者",
                    chat_id="qq:999:group", user_id="111", user_nickname="甲",
                    last_seen="100", interval_minutes=30, enabled=True,
                    created_at=time.time(), last_checked=0.0)
    defaults.update(kw)
    return m.Subscription.create(**defaults)


class TestResolve:
    def test_pixiv_uid_and_url(self, _env):
        assert _env._resolve_pixiv_uid("16689973") == "16689973"
        assert _env._resolve_pixiv_uid("https://www.pixiv.net/users/16689973") == "16689973"
        assert _env._resolve_pixiv_uid("随便什么") == ""

    @pytest.mark.asyncio
    async def test_bili_mid_passthrough(self, _env):
        mid, name = await _env._resolve_bili_mid("473837611")
        assert mid == "473837611"

    @pytest.mark.asyncio
    async def test_bili_search_by_name(self, _env, monkeypatch):
        from junjun_skills.plugins.bilibili import tools as bili

        async def _sign(params):
            return params

        async def _fetch(url, params=None):
            return {"data": {"result": [{"mid": 42, "uname": "<em>测试</em>UP"}]}}

        monkeypatch.setattr(bili, "_wbi_sign", _sign)
        monkeypatch.setattr(bili, "_fetch_json", _fetch)
        mid, name = await _env._resolve_bili_mid("测试UP")
        assert mid == "42" and name == "测试UP"


class TestCheckers:
    @pytest.mark.asyncio
    async def test_pixiv_diff(self, _env, monkeypatch):
        """只报比 last_seen 新的 P 站小说，旧->新排序。"""
        from junjun_skills.plugins.pixiv_novel import tools as pixiv

        async def _works(uid):
            return {"author": "某作者", "series": [], "novels": [
                {"id": "103", "title": "新篇3"},
                {"id": "101", "title": "新篇1"},
                {"id": "99", "title": "旧篇"},   # <= baseline 100 不报（字符串比较转 int）
            ]}

        monkeypatch.setattr(pixiv, "_fetch_author_works", _works)
        sub = SimpleNamespace(target_id="16689973", last_seen="100")
        items, name = await _env._check_pixiv_author(sub)
        assert name == "某作者"
        assert [i["title"] for i in items] == ["新篇1", "新篇3"]
        assert items[-1]["seen"] == "103"

    @pytest.mark.asyncio
    async def test_bili_diff(self, _env, monkeypatch):
        """只报 pubdate 比 last_seen 新的视频。"""
        from junjun_skills.plugins.bilibili import tools as bili

        async def _sign(params):
            return params

        async def _fetch(url, params=None):
            return {"data": {"list": {"vlist": [
                {"bvid": "BV3", "title": "新视频", "created": 2000, "author": "某UP"},
                {"bvid": "BV1", "title": "旧视频", "created": 500, "author": "某UP"},
            ]}}}

        monkeypatch.setattr(bili, "_wbi_sign", _sign)
        monkeypatch.setattr(bili, "_fetch_json", _fetch)
        sub = SimpleNamespace(target_id="42", last_seen="1000")
        items, name = await _env._check_bili_up(sub)
        assert name == "某UP"
        assert [i["title"] for i in items] == ["新视频"]
        assert items[0]["url"].endswith("/BV3")


class TestSubscribeFlow:
    @pytest.mark.asyncio
    async def test_baseline_no_history_spam(self, _env, monkeypatch):
        """订阅当下以最新内容为基线——首次检查不会轰炸历史内容。"""
        from junjun_skills.plugins.pixiv_novel import tools as pixiv

        async def _works(uid):
            return {"author": "某作者", "series": [], "novels": [
                {"id": "105", "title": "最新"}, {"id": "104", "title": "次新"}]}

        monkeypatch.setattr(pixiv, "_fetch_author_works", _works)
        baseline, name = await _env._baseline_for("pixiv_author", "16689973")
        assert baseline == "105" and name == "某作者"

        sub = _mk_sub(_env, last_seen=baseline, last_checked=0)
        sent = []

        async def _notify(s, items):
            sent.append(items)

        monkeypatch.setattr(_env, "_notify", _notify)
        await _env.check_subscriptions()
        assert not sent  # 无新内容不推送

    @pytest.mark.asyncio
    async def test_check_pushes_and_updates_state(self, _env, monkeypatch):
        """有更新：推送 + last_seen/last_checked/target_name 更新。"""
        from junjun_skills.plugins.pixiv_novel import tools as pixiv

        async def _works(uid):
            return {"author": "真名", "series": [], "novels": [{"id": "107", "title": "新"}]}

        monkeypatch.setattr(pixiv, "_fetch_author_works", _works)
        sub = _mk_sub(_env, target_name="", last_seen="100")
        sent = []

        async def _notify(s, items):
            sent.append((s, items))

        monkeypatch.setattr(_env, "_notify", _notify)
        await _env.check_subscriptions()
        assert len(sent) == 1
        fresh = m.Subscription.get_by_id(sub.id)
        assert fresh.last_seen == "107"
        assert fresh.target_name == "真名"
        assert fresh.last_checked > 0

    @pytest.mark.asyncio
    async def test_interval_throttle(self, _env, monkeypatch):
        """未到检查间隔的订阅跳过。"""
        from junjun_skills.plugins.pixiv_novel import tools as pixiv
        called = []

        async def _works(uid):
            called.append(1)
            return {"author": "", "series": [], "novels": [{"id": "107", "title": "新"}]}

        monkeypatch.setattr(pixiv, "_fetch_author_works", _works)
        _mk_sub(_env, last_checked=time.time())  # 刚查过
        monkeypatch.setattr(_env, "_notify", lambda s, i: None)
        await _env.check_subscriptions()
        assert not called


class TestPermission:
    def test_unsub_creator_and_admin(self, _env, monkeypatch):
        sub = _mk_sub(_env, user_id="111")
        monkeypatch.setenv("ADMIN_QQ", "99999")
        # 无关人员不能删
        assert "只有本人或管理员" in _env._do_unsub(str(sub.id), caller="222")
        assert m.Subscription.get_by_id(sub.id).enabled
        # 创建者可以删
        assert "已取消" in _env._do_unsub(str(sub.id), caller="111")
        assert not m.Subscription.get_by_id(sub.id).enabled
        # 管理员可以删别人的
        sub2 = _mk_sub(_env, user_id="111")
        assert "已取消" in _env._do_unsub(str(sub2.id), caller="99999")


class TestNotifyFormat:
    def test_fmt(self, _env):
        sub = SimpleNamespace(kind="bili_up", target_name="某UP", target_id="42")
        text = _env._fmt_notify(sub, [{"seen": "1", "title": "新视频", "url": "http://x"}])
        assert "某UP" in text and "《新视频》" in text and "http://x" in text


class TestDoSubscribe:
    @pytest.mark.asyncio
    async def test_create_via_command_path(self, _env, monkeypatch):
        """确定性命令通道：_do_subscribe 直接落库（绕过 LLM 工具选择）。"""
        from junjun_skills.plugins.pixiv_novel import tools as pixiv

        async def _works(uid):
            return {"author": "某作者", "series": [], "novels": [{"id": "105", "title": "最新"}]}

        monkeypatch.setattr(pixiv, "_fetch_author_works", _works)
        reply = await _env._do_subscribe("pixiv", "16689973", "qq:999:group", "111", "甲")
        assert "订阅好了" in reply and "#" in reply
        rows = list(m.Subscription.select())
        assert len(rows) == 1
        assert rows[0].kind == "pixiv_author" and rows[0].last_seen == "105"
        assert rows[0].target_name == "某作者"

    @pytest.mark.asyncio
    async def test_bad_source_rejected(self, _env):
        reply = await _env._do_subscribe("youtube", "123", "qq:999:group", "111", "甲")
        assert "不认识订阅源" in reply


class TestRemoveWhere:
    def test_remove_matching_memories(self, tmp_path, monkeypatch):
        """remove_where：按谓词删除并重建索引（/forget 的底层）。"""
        from junjun_memory.long_term import LongTermMemory
        ltm = LongTermMemory.__new__(LongTermMemory)
        ltm._dir = tmp_path
        ltm._items = []
        ltm._index = None
        ltm._vec_map = []
        ltm._loaded = True
        import asyncio
        asyncio.run(ltm.add("温衿青让君君盯着作者16689973", "qq:1:group"))
        asyncio.run(ltm.add("白菜兔今天吃了火锅", "qq:1:group"))
        asyncio.run(ltm.add("另一条关于16689973的记录", "qq:1:group"))
        removed = ltm.remove_where(lambda it: "16689973" in it.text)
        assert removed == 2
        assert len(ltm._items) == 1
        assert "火锅" in ltm._items[0].text
