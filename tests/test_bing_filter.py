"""Bing 相关性过滤回归（2026-08-09 宁德→深圳事故）：
连续中文整串当关键词导致「宁德至深圳」「宁德到深圳北」零命中，
5 个查询 3 个被过滤清零。修复：长 CJK 串叠加 bigram + 过滤清零回退未过滤。
"""

from bs4 import BeautifulSoup

from junjun_skills.plugins.google_search.engines.bing import BingEngine


def _engine():
    return BingEngine(config={})


class TestBuildKeywords:
    def test_long_cjk_gets_bigrams(self):
        kws = _engine()._build_keywords("宁德到深圳 高铁 时刻表")
        assert "宁德到深圳" in kws      # 整串保留
        assert "宁德" in kws and "深圳" in kws  # bigram 让城市名独立命中

    def test_short_cjk_not_split(self):
        """短词（<5 字）不拆——「高铁」拆成「高/铁」单字没意义。"""
        kws = _engine()._build_keywords("高铁 票价")
        assert kws == ["高铁", "票价"]

    def test_english_unchanged(self):
        kws = _engine()._build_keywords("python tutorial")
        assert kws == ["python", "tutorial"]

    def test_single_char_dropped(self):
        assert "一" not in _engine()._build_keywords("一 高铁")


class TestRelevance:
    def test_variant_wording_matches_via_bigram(self):
        """事故核心：查询「宁德到深圳」，结果标题「宁德至深圳北」必须命中。"""
        e = _engine()
        kws = e._build_keywords("宁德到深圳 高铁 时刻表 票价 耗时")
        assert e._is_relevant("宁德至深圳北动车时刻表", "", "", kws)

    def test_truly_irrelevant_still_filtered(self):
        """误判回归：完全无关的结果仍被过滤（过滤不是摆设）。"""
        e = _engine()
        kws = e._build_keywords("宁德到深圳 高铁 时刻表")
        assert not e._is_relevant("猫咪日常护理指南", "可爱视频合集", "http://x", kws)


_HTML = """<html><body><ol id="b_results">
<li class="b_algo"><h2><a href="http://a.test/1">完全无关的标题甲</a></h2>
<div class="b_caption"><p>摘要甲</p></div></li>
<li class="b_algo"><h2><a href="http://b.test/2">无关标题乙</a></h2>
<div class="b_caption"><p>摘要乙</p></div></li>
</ol></body></html>"""


class TestEmptyFilterFallback:
    def test_parse_unfiltered_returns_results(self):
        e = _engine()
        soup = BeautifulSoup(_HTML, "html.parser")
        results = e._parse_page_unfiltered(soup)
        assert [r.url for r in results] == ["http://a.test/1", "http://b.test/2"]

    def test_filtered_kills_but_unfiltered_has_goods(self):
        """过滤全灭时未过滤解析仍有货（供链尾兜底使用）。"""
        e = _engine()
        soup = BeautifulSoup(_HTML, "html.parser")
        kws = e._build_keywords("宁德到深圳 高铁 时刻表")
        assert e._parse_page_results(soup, kws) == []       # 过滤全灭
        assert len(e._parse_page_unfiltered(soup)) == 2     # 兜底有货

    def test_search_unfiltered_skips_filter(self, monkeypatch):
        """search_unfiltered：走 CN 变体、不应用关键词过滤。"""
        import pytest
        e = _engine()

        async def fake_get_next_page(query, **kw):
            return _HTML

        monkeypatch.setattr(e, "_get_next_page", fake_get_next_page)

        async def run():
            return await e.search_unfiltered("宁德到深圳 高铁", 5)

        results = __import__("asyncio").run(run())
        assert len(results) == 2
