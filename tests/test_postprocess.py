"""回复后处理单测：splitter / typo / 流水线。"""

import random

from junjun_agent.postprocess.splitter import split_response, typing_delay
from junjun_agent.postprocess.typo import ChineseTypoGenerator
from junjun_agent.postprocess import process_response


class TestSplitter:
    def test_empty(self):
        assert split_response("") == []
        assert split_response("   ") == []

    def test_short_text_single_piece(self):
        out = split_response("好呀", rand=random.Random(1))
        assert out == ["好呀"]

    def test_hard_stop_always_splits(self):
        out = split_response("今天真开心！明天也要加油？好的。", rand=random.Random(1))
        assert len(out) >= 2

    def test_max_sentence_overflow_return_all(self):
        text = "一。二。三。四。五。六。七。"
        out = split_response(text, max_sentence_num=3, enable_overflow_return_all=True,
                             rand=random.Random(1))
        assert len(out) == 1  # 超限整发
        assert "一" in out[0] and "七" in out[0]

    def test_max_sentence_overflow_truncate(self):
        text = "一。二。三。四。五。"
        out = split_response(text, max_sentence_num=2, enable_overflow_return_all=False,
                             rand=random.Random(1))
        assert len(out) == 2

    def test_hard_char_limit(self):
        text = "啊" * 300
        out = split_response(text, max_chars_per_message=100, rand=random.Random(1))
        assert all(len(p) <= 100 for p in out)

    def test_stage_directions_removed(self):
        out = split_response("（摸摸头）你真棒", rand=random.Random(1))
        assert out and "摸摸头" not in "".join(out)
        assert "你真棒" in "".join(out)

    def test_disable_returns_whole(self):
        text = "第一句。第二句。第三句。"
        assert split_response(text, enable=False) == [text]

    def test_deterministic_with_seed(self):
        text = "今天吃了火锅，特别辣，但是很爽，下次还去。"
        a = split_response(text, rand=random.Random(7))
        b = split_response(text, rand=random.Random(7))
        assert a == b

    def test_typing_delay_bounds(self):
        d = typing_delay("一二三四五", rand=random.Random(1))
        assert 0.1 < d < 3.7


class TestTypo:
    def test_url_protected(self):
        gen = ChineseTypoGenerator(error_rate=1.0)  # 100% 替换率压测保护
        text = "看这个 https://example.com/abc 链接"
        out = gen.create_typo_sentence(text, rand=random.Random(1))
        assert "https://example.com/abc" in out

    def test_english_number_protected(self):
        gen = ChineseTypoGenerator(error_rate=1.0)
        out = gen.create_typo_sentence("版本是 v2.5.1 哦", rand=random.Random(1))
        assert "v2.5.1" in out

    def test_at_mention_protected(self):
        gen = ChineseTypoGenerator(error_rate=1.0)
        out = gen.create_typo_sentence("@张三 你好", rand=random.Random(1))
        assert out.startswith("@张三")

    def test_zero_rate_no_change(self):
        gen = ChineseTypoGenerator(error_rate=0.0, word_replace_rate=0.0)
        text = "今天天气真好"
        assert gen.create_typo_sentence(text, rand=random.Random(1)) == text

    def test_high_rate_changes_chinese(self):
        gen = ChineseTypoGenerator(error_rate=1.0, tone_error_rate=0.0, word_replace_rate=0.0)
        text = "今天天气真好呀朋友"
        out = gen.create_typo_sentence(text, rand=random.Random(3))
        assert out != text
        assert len(out) == len(text)  # 同音单字替换不变长


class TestPipeline:
    def test_think_tag_stripped(self):
        out = process_response("<think>内心思考</think>好呀好呀", rand=random.Random(1))
        joined = "".join(m.text for m in out)
        assert "内心思考" not in joined
        assert "好呀" in joined

    def test_nickname_prefix_stripped(self):
        out = process_response("君君: 在呢在呢", rand=random.Random(1))
        assert out[0].text.startswith("在呢")

    def test_empty_returns_nothing(self):
        assert process_response("") == []
        assert process_response("<think>只有思考</think>") == []

    def test_multi_piece_has_delay(self):
        text = "第一句话说完了！第二句话也说完了！第三句话讲完了。"
        out = process_response(text, rand=random.Random(5))
        if len(out) > 1:
            assert out[0].delay == 0.0
            assert all(m.delay > 0 for m in out[1:])
