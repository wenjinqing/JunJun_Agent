"""adapter 感知增强测试：@ 真实昵称解析 + 引用消息内容展开。"""

import importlib

import pytest

# 注意：recv_handler/__init__ 导出了同名 message_handler 单例，
# from ... import message_handler 拿到的是实例而非模块，必须显式加载模块对象
mh = importlib.import_module("junjun_adapter_napcat.recv_handler.message_handler")


@pytest.fixture(autouse=True)
def _clear_cache():
    mh._NICK_CACHE.clear()
    yield
    mh._NICK_CACHE.clear()


@pytest.fixture
def _mock_nc(monkeypatch):
    """拦截 NapCat API 调用。"""
    calls = []

    async def _send(action, params):
        calls.append((action, params))
        if action == "get_group_member_info":
            return {"data": {"card": "白菜兔", "nickname": "白菜"}}
        if action == "get_msg":
            return {"data": {
                "sender": {"nickname": "阿鹤", "card": "鹤"},
                "message": [
                    {"type": "text", "data": {"text": "今晚开黑吗"}},
                    {"type": "image", "data": {}},
                ],
            }}
        return {"status": "error"}

    import junjun_adapter_napcat.send_handler.nc_sending as nc
    monkeypatch.setattr(nc.nc_message_sender, "send_message_to_napcat", _send)
    return calls


def _handler():
    return mh.MessageHandler.__new__(mh.MessageHandler)


class TestNicknameResolution:
    @pytest.mark.asyncio
    async def test_at_resolves_to_card(self, _mock_nc):
        segs, at_bot = await _handler()._parse_message_segments(
            [{"type": "at", "data": {"qq": "111"}}, {"type": "text", "data": {"text": "在吗"}}],
            self_id="999", group_id="12345")
        assert segs[0].data == "@白菜兔 "  # card 优先
        assert at_bot is False

    @pytest.mark.asyncio
    async def test_at_bot_shows_as_you(self, _mock_nc):
        segs, at_bot = await _handler()._parse_message_segments(
            [{"type": "at", "data": {"qq": "999"}}], self_id="999", group_id="12345")
        assert segs[0].data == "@你 "
        assert at_bot is True

    @pytest.mark.asyncio
    async def test_nickname_cached(self, _mock_nc):
        h = _handler()
        await h._parse_message_segments([{"type": "at", "data": {"qq": "111"}}],
                                        self_id="999", group_id="12345")
        await h._parse_message_segments([{"type": "at", "data": {"qq": "111"}}],
                                        self_id="999", group_id="12345")
        api_calls = [c for c in _mock_nc if c[0] == "get_group_member_info"]
        assert len(api_calls) == 1  # 第二次命中缓存

    @pytest.mark.asyncio
    async def test_no_group_degrades(self, _mock_nc):
        segs, _ = await _handler()._parse_message_segments(
            [{"type": "at", "data": {"qq": "111"}}], self_id="999", group_id="")
        assert segs[0].data == "@某人 "
        assert not _mock_nc  # 私聊不查群成员

    @pytest.mark.asyncio
    async def test_api_failure_degrades(self, monkeypatch):
        async def _err(action, params):
            raise RuntimeError("network down")

        import junjun_adapter_napcat.send_handler.nc_sending as nc
        monkeypatch.setattr(nc.nc_message_sender, "send_message_to_napcat", _err)
        segs, _ = await _handler()._parse_message_segments(
            [{"type": "at", "data": {"qq": "111"}}], self_id="999", group_id="12345")
        assert segs[0].data == "@某人 "


class TestReplyResolution:
    @pytest.mark.asyncio
    async def test_reply_expanded(self, _mock_nc):
        segs, _ = await _handler()._parse_message_segments(
            [{"type": "reply", "data": {"id": "777"}},
             {"type": "text", "data": {"text": "不来"}}],
            self_id="999", group_id="12345")
        assert segs[0].type == "text"
        assert segs[0].data == "[回复 鹤: 今晚开黑吗[图片]]"

    @pytest.mark.asyncio
    async def test_reply_failure_degrades(self, monkeypatch):
        async def _err(action, params):
            return {"status": "error"}

        import junjun_adapter_napcat.send_handler.nc_sending as nc
        monkeypatch.setattr(nc.nc_message_sender, "send_message_to_napcat", _err)
        segs, _ = await _handler()._parse_message_segments(
            [{"type": "reply", "data": {"id": "777"}}], self_id="999", group_id="12345")
        assert segs[0].data == "[回复某条消息]"

    @pytest.mark.asyncio
    async def test_reply_long_truncated(self, monkeypatch):
        async def _send(action, params):
            return {"data": {"sender": {"nickname": "甲"},
                             "message": [{"type": "text", "data": {"text": "长" * 300}}]}}

        import junjun_adapter_napcat.send_handler.nc_sending as nc
        monkeypatch.setattr(nc.nc_message_sender, "send_message_to_napcat", _send)
        segs, _ = await _handler()._parse_message_segments(
            [{"type": "reply", "data": {"id": "1"}}], self_id="999", group_id="12345")
        assert len(segs[0].data) <= 215  # 「[回复 甲: 」+ 200 + … + 」
