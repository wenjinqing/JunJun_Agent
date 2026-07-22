"""数据契约与短期记忆单测。"""

from junjun_core.contracts import ReplySet, ReplySegment
from junjun_memory.short_term import ShortTermMemory


class TestReplySet:
    def test_single_text_segment(self):
        rs = ReplySet(target_group_id="999", segments=[ReplySegment("text", "hi")])
        mb = rs.to_message_base()
        assert mb.message_segment.type == "text"
        assert mb.message_segment.data == "hi"
        assert mb.message_info.group_info.group_id == "999"

    def test_reply_quote_makes_seglist(self):
        rs = ReplySet(
            target_group_id="999",
            segments=[ReplySegment("text", "hi")],
            reply_to_message_id="12345",
        )
        seg = rs.to_message_base().message_segment
        assert seg.type == "seglist"
        assert seg.data[0].type == "reply"
        assert seg.data[0].data == "12345"
        assert seg.data[1].type == "text"

    def test_private_no_group_info(self):
        rs = ReplySet(target_user_id="777", segments=[ReplySegment("text", "x")])
        mb = rs.to_message_base()
        assert mb.message_info.group_info is None
        assert mb.message_info.user_info.user_id == "777"


class TestShortTermMemory:
    def test_window_trim(self):
        m = ShortTermMemory(max_size=3)
        for i in range(5):
            m.add_user(f"msg{i}", "甲")
        assert len(m.entries) == 3
        assert m.entries[0].text == "msg2"

    def test_render_with_nickname(self):
        m = ShortTermMemory()
        m.add_user("你好", "甲", at_bot=False)
        m.add_user("君君在吗", "乙", at_bot=True)
        m.add_bot("在的")
        # 默认 bot 不进 context（防复读）
        text = m.render()
        assert "甲: 你好" in text
        assert "乙 [@你]: 君君在吗" in text
        assert "你: 在的" not in text
        # include_bot=True 时 bot 进 context（调试用）
        text_with_bot = m.render(include_bot=True)
        assert "你: 在的" in text_with_bot

    def test_render_limit(self):
        m = ShortTermMemory()
        for i in range(10):
            m.add_user(f"m{i}", "甲")
        assert "m9" in m.render(limit=2)
        assert "m7" not in m.render(limit=2)

    def test_last_user_entry_skips_bot(self):
        m = ShortTermMemory()
        m.add_user("q", "甲", message_id="42")
        m.add_bot("a")
        assert m.last_user_entry().message_id == "42"
