"""回复后处理优化测试：切分点丢逗号/句号 + markdown 标记清理。"""

import random

from junjun_agent.postprocess.cleaner import clean_markdown
from junjun_agent.postprocess.splitter import split_response


class TestStripSplitPunct:
    def test_hard_stop_period_dropped(self):
        """句号切分点：气泡不带句号。"""
        out = split_response("好的。我知道了。明天见", rand=random.Random(1))
        assert all(not p.endswith("。") for p in out)
        assert "".join(out).replace("，", "") .replace("。", "") != ""
        assert "明天见" in out[-1]

    def test_exclaim_question_kept(self):
        """感叹号/问号保留。"""
        out = split_response("太好了！真的吗？是啊", rand=random.Random(1))
        joined = "".join(out)
        assert "！" in joined and "？" in joined

    def test_comma_split_drops_comma(self):
        """逗号处切开时不带逗号（强制不合并）。"""
        class _NeverMerge:
            def random(self): return 1.0  # > merge_p -> 必切

        out = split_response("好的，我知道了，马上来", rand=_NeverMerge())
        assert out == ["好的", "我知道了", "马上来"]

    def test_comma_merged_keeps_comma(self):
        """概率合并时逗号在句中保留。"""
        class _AlwaysMerge:
            def random(self): return 0.0  # <= merge_p -> 必合并

        out = split_response("好的，我知道了", rand=_AlwaysMerge())
        assert out == ["好的，我知道了"]

    def test_ellipsis_not_stripped(self):
        """省略号 ... 不误伤。"""
        out = split_response("这个嘛...", rand=random.Random(1))
        assert out == ["这个嘛..."]

    def test_decimal_not_stripped(self):
        """小数点不误伤。"""
        out = split_response("评分3.5", rand=random.Random(1))
        assert out == ["评分3.5"]

    def test_strip_disabled_keeps_punct(self):
        """strip_split_punct=False 回到旧行为（标点保留）。"""
        class _NeverMerge:
            def random(self): return 1.0

        out = split_response("好的，我知道了。", rand=_NeverMerge(), strip_split_punct=False)
        assert out[0].endswith("，") and out[1].endswith("。")

    def test_overflow_return_all_keeps_punct(self):
        """超上限整发：保留原文标点。"""
        text = "一。二。三。四。五。六。七。"
        out = split_response(text, max_sentence_num=3, enable_overflow_return_all=True,
                             rand=random.Random(1))
        assert out == [text]


class TestCleanMarkdown:
    def test_bold(self):
        assert clean_markdown("这是**重点**内容") == "这是重点内容"

    def test_bold_underscore(self):
        assert clean_markdown("这是__重点__内容") == "这是重点内容"

    def test_italic(self):
        assert clean_markdown("这是*强调*内容") == "这是强调内容"

    def test_heading(self):
        assert clean_markdown("## 我的想法\n正文") == "我的想法\n正文"

    def test_inline_code(self):
        assert clean_markdown("用 `pip` 安装") == "用 pip 安装"

    def test_math_not_touched(self):
        """算式里的星号（内容无字母/中文）不误伤。"""
        assert clean_markdown("3*5*6=90") == "3*5*6=90"

    def test_unpaired_star_kept(self):
        assert clean_markdown("3*5=15") == "3*5=15"

    def test_kaomoji_protected(self):
        """颜文字内的 * 不被误删。"""
        assert clean_markdown("(*/ω\\*) 害羞") == "(*/ω\\*) 害羞"

    def test_no_markdown_passthrough(self):
        assert clean_markdown("普通消息，没有标记。") == "普通消息，没有标记。"
