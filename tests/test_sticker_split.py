"""图片/表情包区分测试：OneBot sub_type=1 与 mface 是表情包（可偷），普通图片不偷。"""

import pytest
from maim_message import Seg

from junjun_core.gateway.router import (
    _extract_images, _extract_stickers, _extract_text, _has_emoji,
)


def _seg(*subs):
    return Seg(type="seglist", data=list(subs))


class TestStickerVsImage:
    def test_image_and_sticker_split(self):
        seg = _seg(Seg(type="text", data="看"),
                   Seg(type="image", data="http://x/photo.jpg"),
                   Seg(type="sticker", data="http://x/sticker.png"))
        assert _extract_images(seg) == ["http://x/photo.jpg"]
        assert _extract_stickers(seg) == ["http://x/sticker.png"]

    def test_text_placeholders(self):
        seg = _seg(Seg(type="image", data="http://x/a.jpg"),
                   Seg(type="sticker", data="http://x/b.png"))
        assert _extract_text(seg) == "[图片][表情]"

    def test_sticker_counts_as_emoji(self):
        assert _has_emoji(_seg(Seg(type="sticker", data="http://x/b.png"))) is True
        assert _has_emoji(_seg(Seg(type="image", data="http://x/a.jpg"))) is False

    def test_empty_sticker_data_ignored(self):
        seg = _seg(Seg(type="sticker", data=""))
        assert _extract_stickers(seg) == []


class TestAdapterSegMapping:
    @pytest.mark.asyncio
    async def test_image_sub_type_and_mface(self):
        from junjun_adapter_napcat.recv_handler.message_handler import MessageHandler
        handler = MessageHandler.__new__(MessageHandler)  # 只测纯解析，不触发发送
        segs, at_bot = await handler._parse_message_segments([
            {"type": "image", "data": {"url": "http://x/photo.jpg", "sub_type": 0}},
            {"type": "image", "data": {"url": "http://x/sticker.png", "sub_type": 1}},
            {"type": "mface", "data": {"url": "http://x/mall.gif"}},
        ])
        types = [(s.type, s.data) for s in segs]
        assert ("image", "http://x/photo.jpg") in types
        assert ("sticker", "http://x/sticker.png") in types
        assert ("sticker", "http://x/mall.gif") in types
        assert at_bot is False
