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
    def test_keyword_search(self, tmp_path, monkeypatch):
        import peewee
        from junjun_core.database import models as m
        db = peewee.SqliteDatabase(str(tmp_path / "t.db"))
        with db.bind_ctx([m.Messages]):
            db.create_tables([m.Messages])
            m.Messages.create(chat_id="qq:1:group", user_nickname="甲", time=1.0, message_id="m1",
                              processed_plain_text="今晚吃火锅吗", bot_id="10000001")
            m.Messages.create(chat_id="qq:1:group", user_nickname="乙", time=2.0, message_id="m2",
                              processed_plain_text="吃过了", bot_id="10000001")

            from junjun_skills.builtin.memory_skills import current_chat_id
            current_chat_id.set("qq:1:group")
            from junjun_skills.builtin.action_skills import query_chat_history
            result = query_chat_history.invoke({"keyword": "火锅"})
            assert "火锅" in result and "甲" in result
            assert "吃过了" not in result
            empty = query_chat_history.invoke({"keyword": "不存在词"})
            assert "没有找到" in empty


class TestNoticePoke:
    @pytest.mark.asyncio
    async def test_poke_to_bot_becomes_addressed_message(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler import notice_handler as nh

        sent = []

        class _FakeSend:
            async def message_send(self, msg_base):
                sent.append(msg_base)

        monkeypatch.setattr(nh, "message_send_instance", _FakeSend())

        async def _allow(u, g):
            return True

        monkeypatch.setattr(nh, "message_handler_allow", _allow)

        import junjun_adapter_napcat.config as cfg_mod

        class _Cfg:
            class maibot_server:
                platform_name = "qq"

        monkeypatch.setattr(cfg_mod, "get_config", lambda: _Cfg())

        await nh.notice_handler.handle_notice({
            "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 10000001, "user_id": 12345, "group_id": 999,
        })
        assert len(sent) == 1
        msg = sent[0]
        assert msg.message_info.additional_config["at_bot"] is True
        assert "戳" in msg.message_segment.data

    @pytest.mark.asyncio
    async def test_poke_not_targeting_bot_ignored(self, monkeypatch):
        from junjun_adapter_napcat.recv_handler import notice_handler as nh
        sent = []

        class _FakeSend:
            async def message_send(self, msg_base):
                sent.append(msg_base)

        monkeypatch.setattr(nh, "message_send_instance", _FakeSend())
        await nh.notice_handler.handle_notice({
            "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "self_id": 10000001, "target_id": 22222, "user_id": 12345, "group_id": 999,
        })
        assert sent == []


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
