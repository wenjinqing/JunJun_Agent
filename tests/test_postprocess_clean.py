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


class TestCleanTone:
    """语气词/emoji 结构层清理（2026-08-17 用户拍板：回复不要语气词、不要
    emoji——reply_style 早写「不用 emoji」弱模型照吐，prompt 是劝告，
    这里是闸门）。剥句末位置 + 语义词保留，误判回归是主体。"""

    def test_emoji_stripped(self):
        from junjun_agent.postprocess.cleaner import clean_tone
        assert clean_tone("今晚真开心🥳🎉") == "今晚真开心"
        assert clean_tone("好🥳耶") == "好耶"              # 句中也剥
        assert clean_tone("⭐⭐⭐好评✅") == "好评"
        out = clean_tone("第一 1️⃣ 步")                    # 键帽序列
        assert "1" in out and "️⃣" not in out

    def test_zwj_sequence_stripped(self):
        from junjun_agent.postprocess.cleaner import clean_tone
        assert clean_tone("看👨‍👩‍👧这个") == "看这个"

    def test_trailing_particles_stripped(self):
        from junjun_agent.postprocess.cleaner import clean_tone
        assert clean_tone("好的呀。") == "好的。"
        assert clean_tone("知道啦~") == "知道"
        assert clean_tone("在呢") == "在"
        assert clean_tone("乖~") == "乖"
        assert clean_tone("画好马上发你，这回真不糊弄你啦") == "画好马上发你，这回真不糊弄你"
        assert clean_tone("好滴～～") == "好滴"

    def test_multiline_per_line(self):
        from junjun_agent.postprocess.cleaner import clean_tone
        out = clean_tone("在呢在呢\n马上画好啦！")
        assert out == "在呢在\n马上画好！"       # 每行行尾都算句末位置

    def test_semantic_particles_kept(self):
        """语义词不剥：吗/么/吧/的/了——剥了疑问句变陈述句。"""
        from junjun_agent.postprocess.cleaner import clean_tone
        for t in ("你吃饭了吗", "你吃饭了吗？", "走吧", "好吧",
                  "好的", "我知道了", "吃了么"):
            assert clean_tone(t) == t, t

    def test_word_collision_kept(self):
        """构词误伤回归：唢呐/嘻哈/哎哟/笑声本体不是语气词。"""
        from junjun_agent.postprocess.cleaner import clean_tone
        for t in ("他会吹唢呐", "哈哈哈", "嘻哈文化", "哎哟", "笑死",
                  "哈哈哈哈草"):
            assert clean_tone(t) == t, t

    def test_mid_sentence_particle_kept(self):
        """逗号前的「他呢」「这个嘛」是话题标记，剥了语句不通——只剥句末。"""
        from junjun_agent.postprocess.cleaner import clean_tone
        assert clean_tone("他呢，我不太熟") == "他呢，我不太熟"
        assert clean_tone("在呢在呢，别催啦。") == "在呢在呢，别催。"

    def test_empty_line_guard(self):
        """剥完行里不剩字（「啊！」->「！」）就回滚——空感叹号比语气词更怪。"""
        from junjun_agent.postprocess.cleaner import clean_tone
        assert clean_tone("啊！") == "啊！"

    def test_switches(self):
        from junjun_agent.postprocess.cleaner import clean_tone
        t = "好呀。🥳"
        # emoji 关/语气词开：emoji 保留，呀照剥（标点锚定句末位）
        assert clean_tone(t, strip_emoji=False) == "好。🥳"
        # 语气词关/emoji 开：呀保留，emoji 照剥
        assert clean_tone(t, strip_particles=False) == "好呀。"
        assert clean_tone(t, strip_emoji=False, strip_particles=False) == t

    def test_pipeline_integration(self, _fake_bot_config):
        """流水线级：process_response 默认开清理，关开关则原样过。"""
        from junjun_agent.postprocess import process_response
        _fake_bot_config.raw["chinese_typo"] = {"enable": False}
        _fake_bot_config.raw["response_splitter"] = {"enable": False}
        out = process_response("好的，这就去呀~🥳")
        assert len(out) == 1
        assert out[0].text == "好的，这就去"     # 句尾呀/波浪号/emoji 全剥
        # 关掉两个开关：原样直出
        _fake_bot_config.raw["response_post_process"] = {
            "strip_emoji": False, "strip_tone_particles": False}
        out2 = process_response("好的呀🥳")
        assert out2[0].text == "好的呀🥳"
