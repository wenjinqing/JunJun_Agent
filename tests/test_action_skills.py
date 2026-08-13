"""动作类 skill + poke 入站 + 合并转发展开 测试。"""

import pytest

from junjun_skills import registry


class TestActionSkillRegistration:
    def test_builtin_loads_all(self):
        registry.load_builtin()
        names = {t.name for t in registry.get_tools()}
        expected = {
            "get_time", "do_not_reply",
            "recall_memory", "save_memory", "manage_user_profile", "query_jargon", "learn_jargon",
            "set_reminder", "list_reminders", "cancel_reminder_task", "manage_mood",
            "send_emoji", "search_knowledge", "import_knowledge",
            "send_message", "send_poke", "get_weather", "query_chat_history",
            "peek_group_chat",
        }
        assert expected <= names

    def test_schemas_valid(self):
        registry.load_builtin()
        for t in registry.get_tools():
            assert t.args is not None  # args_schema 合法可生成


class TestSendPoke:
    @pytest.mark.asyncio
    async def test_poke_disabled_by_config(self, _fake_bot_config, monkeypatch):
        from junjun_skills.builtin.action_skills import send_poke
        _fake_bot_config.raw["chat"]["enable_poke"] = False
        result = await send_poke.ainvoke({"user_id": "12345"})
        assert "关闭" in result

    @pytest.mark.asyncio
    async def test_poke_sends_poke_segment(self, monkeypatch):
        from junjun_skills.builtin.action_skills import send_poke
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:999:group")

        sent = []

        class _FakeGW:
            async def send_reply(self, reply_set):
                sent.append(reply_set)

        import junjun_core.gateway.router as router_mod
        monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
        result = await send_poke.ainvoke({"user_id": "12345"})
        assert "12345" in result
        assert sent and sent[0].segments[0].type == "poke"
        assert sent[0].segments[0].data == "12345"
        assert sent[0].target_group_id == "999"


class TestQueryChatHistory:
    def _db(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        return db, m

    def test_keyword_search(self, tmp_path, monkeypatch):
        import time as _t
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            m.Messages.create(chat_id="qq:1:group", user_nickname="甲", time=_t.time(), message_id="m1",
                              processed_plain_text="今晚吃火锅吗", bot_id="10000001")
            m.Messages.create(chat_id="qq:1:group", user_nickname="乙", time=_t.time(), message_id="m2",
                              processed_plain_text="吃过了", bot_id="10000001")

            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:1:group")
            from junjun_skills.builtin.action_skills import query_chat_history, _SEARCH_LOG
            _SEARCH_LOG.clear()
            result = query_chat_history.invoke({"keyword": "火锅"})
            assert "火锅" in result and "甲" in result
            assert "吃过了" not in result
            empty = query_chat_history.invoke({"keyword": "不存在词"})
            assert "没有找到" in empty
            _SEARCH_LOG.clear()

    def test_user_and_days_filter(self, tmp_path, monkeypatch):
        """user 只看某人发的；days 窗口过滤老消息，days=0 搜全部历史。"""
        import time as _t
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            now = _t.time()
            m.Messages.create(chat_id="qq:1:group", user_nickname="甲", user_id="111",
                              time=now, message_id="m1",
                              processed_plain_text="店名叫老码头", bot_id="10000001")
            m.Messages.create(chat_id="qq:1:group", user_nickname="乙", user_id="222",
                              time=now, message_id="m2",
                              processed_plain_text="店名我忘了", bot_id="10000001")
            m.Messages.create(chat_id="qq:1:group", user_nickname="丙", user_id="333",
                              time=now - 40 * 86400, message_id="m3",
                              processed_plain_text="店名是陈年老店", bot_id="10000001")

            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:1:group")
            from junjun_skills.builtin.action_skills import query_chat_history, _SEARCH_LOG
            _SEARCH_LOG.clear()
            # user 过滤：只剩甲
            r = query_chat_history.invoke({"keyword": "店名", "user": "甲"})
            assert "老码头" in r and "我忘了" not in r
            # 默认 30 天窗口：40 天前的被滤掉
            r = query_chat_history.invoke({"keyword": "店名"})
            assert "老码头" in r and "陈年老店" not in r
            # days=0 全历史
            r = query_chat_history.invoke({"keyword": "店名", "days": 0})
            assert "陈年老店" in r
            _SEARCH_LOG.clear()

    def test_limit_capped_at_8(self, tmp_path, monkeypatch):
        import time as _t
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            for i in range(12):
                m.Messages.create(chat_id="qq:1:group", user_nickname="甲",
                                  time=_t.time() + i, message_id=f"m{i}",
                                  processed_plain_text=f"关键词 第{i}条", bot_id="10000001")
            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:1:group")
            from junjun_skills.builtin.action_skills import query_chat_history, _SEARCH_LOG
            _SEARCH_LOG.clear()
            r = query_chat_history.invoke({"keyword": "关键词", "limit": 50})
            assert r.count("\n- ") <= 8
            _SEARCH_LOG.clear()

    def test_rate_limit(self, tmp_path, monkeypatch):
        import time as _t
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:1:group")
            from junjun_skills.builtin.action_skills import query_chat_history, _SEARCH_LOG
            _SEARCH_LOG.clear()
            for _ in range(5):
                query_chat_history.invoke({"keyword": "x"})
            r = query_chat_history.invoke({"keyword": "x"})
            assert "歇会儿" in r
            _SEARCH_LOG.clear()

    def test_privacy_current_chat_only(self, tmp_path, monkeypatch):
        """群里搜不到私聊记录（隐私边界回归）。"""
        import time as _t
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            m.Messages.create(chat_id="qq:111:private", user_nickname="甲",
                              time=_t.time(), message_id="m1",
                              processed_plain_text="私聊的秘密店名", bot_id="10000001")
            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:1:group")
            from junjun_skills.builtin.action_skills import query_chat_history, _SEARCH_LOG
            _SEARCH_LOG.clear()
            r = query_chat_history.invoke({"keyword": "店名", "days": 0})
            assert "没有找到" in r and "秘密" not in r
            _SEARCH_LOG.clear()


class TestPeekGroupChat:
    """跨群围观（2026-08-03 trace：私聊问「其他群在聊什么」被答「看不到没打通」）。
    私聊限定 + 只读群聊（私聊记录永远拿不到）+ 限流。"""

    def _db(self, tmp_path):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        return db, m

    def _seed(self, m):
        import time as _t
        now = _t.time()
        m.Messages.create(chat_id="qq:100:group", group_id="100", user_nickname="甲",
                          time=now - 30, message_id="a1",
                          processed_plain_text="A群在聊新番", bot_id="10000001")
        m.Messages.create(chat_id="qq:100:group", group_id="100", user_nickname="",
                          is_bot=True, time=now - 20, message_id="a2",
                          processed_plain_text="我也在看", bot_id="10000001")
        m.Messages.create(chat_id="qq:200:group", group_id="200", user_nickname="乙",
                          time=now - 10, message_id="b1",
                          processed_plain_text="B群约饭", bot_id="10000001")
        m.Messages.create(chat_id="qq:111:private", group_id="", user_nickname="丙",
                          time=now, message_id="p1",
                          processed_plain_text="私聊秘密暗号", bot_id="10000001")

    def _invoke(self, group=""):
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:111:private")
        from junjun_skills.builtin.action_skills import peek_group_chat, _PEEK_LOG
        _PEEK_LOG.clear()
        return peek_group_chat.invoke({"group": group})

    def test_overview_lists_groups_no_private(self, tmp_path):
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            self._seed(m)
            out = self._invoke("")
            assert "群 100" in out and "群 200" in out
            assert "A群在聊新番" in out and "B群约饭" in out
            assert "「我」: 我也在看" in out  # bot 群回复可见（公开内容）
            assert "私聊秘密暗号" not in out  # 隐私铁律：私聊永远拿不到

    def test_specific_group(self, tmp_path):
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            self._seed(m)
            out = self._invoke("100")
            assert "A群在聊新番" in out and "B群约饭" not in out
            assert "私聊秘密暗号" not in out

    def test_unknown_group_guidance(self, tmp_path):
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            self._seed(m)
            out = self._invoke("999")
            assert "没找到" in out and "群号" in out  # 引导先概览拿群号

    def test_rate_limit(self, tmp_path):
        db, m = self._db(tmp_path)
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            self._seed(m)
            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:111:private")
            from junjun_skills.builtin.action_skills import peek_group_chat, _PEEK_LOG
            _PEEK_LOG.clear()
            for _ in range(3):
                peek_group_chat.invoke({"group": ""})
            assert "歇会儿" in peek_group_chat.invoke({"group": ""})
            _PEEK_LOG.clear()

    def test_private_only_gate(self):
        """群聊场景不绑这个工具（A 群的事不在 B 群说）。"""
        from types import SimpleNamespace
        registry.load_builtin()
        gate = registry._availability.get("peek_group_chat")
        assert gate is not None
        assert gate(SimpleNamespace(is_group=False)) is True
        assert gate(SimpleNamespace(is_group=True)) is False


class TestNoticePoke:
    """戳一戳新政（2026-08-13 用户裁决）：群戳不进决策链（0 token 廉价回敬，
    日额度 3 次），私聊维持「合成 addressed 文本进决策」。"""

    def _common_patches(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler import notice_handler as nh
        nh._reset_for_test()

        sent = []

        async def _fake_send(msg_base):
            sent.append(msg_base)

        # poke 进决策必须经 message_send_instance 走 WS 到核心网关——
        # adapter 是独立进程，本进程的 gateway 只是 echo 占位（旧路由 poke 必丢）
        monkeypatch.setattr(
            "junjun_adapter_napcat.message_sending.message_send_instance.message_send",
            _fake_send)

        async def _allow(u, g):
            return True

        monkeypatch.setattr(nh, "message_handler_allow", _allow)

        import junjun_adapter_napcat.config as cfg_mod

        class _Cfg:
            class junjun_server:
                platform_name = "qq"

        monkeypatch.setattr(cfg_mod, "get_config", lambda: _Cfg())

        nc_calls = []

        class _FakeNC:
            async def send_message_to_napcat(self, action, params):
                nc_calls.append((action, params))
                return {"status": "ok"}

        monkeypatch.setattr(
            "junjun_adapter_napcat.send_handler.nc_sending.nc_message_sender",
            _FakeNC())
        return nh, sent, nc_calls

    @pytest.mark.asyncio
    async def test_private_poke_becomes_addressed_message(self, monkeypatch):
        """私聊戳：合成 addressed 文本走决策链（维持原样，仅私聊）。"""
        nh, sent, nc_calls = self._common_patches(monkeypatch)
        await nh.notice_handler.handle_notice({
            "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 10000001, "user_id": 12345,
        })
        assert len(sent) == 1
        msg = sent[0]
        assert msg.message_info.additional_config["at_bot"] is True
        assert "戳" in msg.message_segment.data
        assert nc_calls == []                      # 私聊不走 adapter 本地回敬

    @pytest.mark.asyncio
    async def test_group_poke_cheap_reply_no_decision(self, monkeypatch):
        """群戳：不进决策（0 token），adapter 本地反戳或发表情。"""
        nh, sent, nc_calls = self._common_patches(monkeypatch)
        await nh.notice_handler.handle_notice({
            "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 10000001, "user_id": 12345, "group_id": 999,
        })
        assert sent == []                          # 决策链一口都没吃到
        assert len(nc_calls) == 1
        action, params = nc_calls[0]
        assert action in ("send_poke", "send_group_msg")
        if action == "send_group_msg":             # 表情路径：内置小黄豆
            assert params["message"][0]["type"] == "face"
        else:                                      # 反戳路径：目标正确
            assert params["user_id"] == 12345 and params["group_id"] == 999

    @pytest.mark.asyncio
    async def test_group_poke_daily_budget(self, monkeypatch):
        """日额度 3 次：第 4 次起直接无视（token 止损的命门）。"""
        nh, sent, nc_calls = self._common_patches(monkeypatch)
        monkeypatch.setattr(nh, "_poke_cfg", lambda: (0, 0, 5, 600, 3))
        poke = {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
                "self_id": 10000001, "target_id": 10000001, "user_id": 12345,
                "group_id": 999}
        for _ in range(5):
            await nh.notice_handler.handle_notice(poke)
        assert sent == []
        assert len(nc_calls) == 3                  # 前 3 次回敬，第 4/5 次无视

    @pytest.mark.asyncio
    async def test_group_poke_throttle_before_budget(self, monkeypatch):
        """连戳先被防抖抑制（60s 窗口），不烧日额度。"""
        nh, sent, nc_calls = self._common_patches(monkeypatch)
        poke = {"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
                "self_id": 10000001, "target_id": 10000001, "user_id": 12345,
                "group_id": 999}
        await nh.notice_handler.handle_notice(poke)
        await nh.notice_handler.handle_notice(poke)   # 60s 内第二戳 -> 抑制
        assert len(nc_calls) == 1

    @pytest.mark.asyncio
    async def test_poke_not_targeting_bot_ignored(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler import notice_handler as nh
        handled = []

        class _FakeGateway:
            async def handle_inbound(self, msg_dict):
                handled.append(msg_dict)

        monkeypatch.setattr("junjun_core.gateway.router._gateway", _FakeGateway())
        await nh.notice_handler.handle_notice({
            "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 22222, "user_id": 12345, "group_id": 999,
        })
        assert handled == []


class TestForwardExpand:
    @pytest.mark.asyncio
    async def test_expand_and_truncate(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler.message_handler import MessageHandler
        import junjun_adapter_napcat.send_handler.nc_sending as nc_mod

        class _FakeNC:
            async def send_message_to_napcat(self, action, params):
                assert action == "get_forward_msg"
                return {"status": "ok", "data": {"message": [
                    {"sender": {"nickname": "甲"}, "message": [{"type": "text", "data": {"text": "x" * 300}}]},
                    {"sender": {"nickname": "乙"}, "message": [{"type": "text", "data": {"text": "y" * 300}}]},
                    {"sender": {"nickname": "丙"}, "message": [{"type": "text", "data": {"text": "z"}}]},
                ]}}

        monkeypatch.setattr(nc_mod, "nc_message_sender", _FakeNC())
        h = MessageHandler()
        text = await h._expand_forward({"id": "abc"})
        assert text.startswith("[合并转发]")
        assert "甲" in text and "乙" in text
        assert "截断" in text  # 300+300 超 500 字截断

    @pytest.mark.asyncio
    async def test_expand_failure_degrades(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler.message_handler import MessageHandler
        import junjun_adapter_napcat.send_handler.nc_sending as nc_mod

        class _FakeNC:
            async def send_message_to_napcat(self, action, params):
                raise RuntimeError("boom")

        monkeypatch.setattr(nc_mod, "nc_message_sender", _FakeNC())
        h = MessageHandler()
        assert await h._expand_forward({"id": "abc"}) == "[合并转发消息]"
