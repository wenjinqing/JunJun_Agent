"""口播文本工程测试：URL/占位符/markdown/emoji/颜文字/舞台指示/重复标点清洗。"""

from junjun_skills.plugins.tts.speakable import make_speakable


class TestSpeakable:
    def test_url_and_placeholder_removed(self):
        out = make_speakable("你看这个 https://b23.tv/abc123 [图片] 超好笑")
        assert "http" not in out and "[图片]" not in out
        assert "超好笑" in out

    def test_markdown_removed(self):
        out = make_speakable("*才不* `关心` #你呢#")
        assert "*" not in out and "`" not in out and "#" not in out
        assert "关心" in out

    def test_emoji_removed(self):
        out = make_speakable("今天天气不错☀️出去走走吧")
        assert "☀" not in out
        assert "出去走走吧" in out

    def test_repeat_punct_collapsed(self):
        assert "！！！" not in make_speakable("太好笑了！！！")
        assert make_speakable("太好笑了！！！").endswith("！")

    def test_stage_direction_removed(self):
        out = make_speakable("（笑）杂鱼就是杂鱼")
        assert "（笑）" not in out
        assert "杂鱼就是杂鱼" in out

    def test_newline_becomes_pause(self):
        out = make_speakable("第一行\n第二行")
        assert "\n" not in out
        assert "第一行，第二行" == out

    def test_empty_after_clean(self):
        assert make_speakable("https://x.com [图片]") == ""
        assert make_speakable("") == ""

    def test_truncation(self):
        long_text = "啊" * 400
        assert len(make_speakable(long_text)) == 300

    def test_normal_text_untouched(self):
        assert make_speakable("今晚一起吃饭吧") == "今晚一起吃饭吧"
