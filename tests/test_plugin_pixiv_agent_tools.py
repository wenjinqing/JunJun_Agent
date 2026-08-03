"""pixiv 插件 LLM 工具面测试（2026-08-03）：

双通道策略——搜索/推荐群私通用；下载/发送仅私聊（群聊给解释性拒绝）。
"""

import time

import pytest

import junjun_skills.plugins.pixiv.agent_tools as at
import junjun_skills.plugins.pixiv.illust as illust
import junjun_skills.plugins.pixiv.novel as novel_mod
from junjun_skills.builtin.memory_skills import current_chat_id


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(at, "_cookie", lambda: "PHPSESSID=1_x")
    at._dl_last.clear()
    token = current_chat_id.set("qq:12345:private")
    yield
    current_chat_id.reset(token)
    at._dl_last.clear()


def _group():
    current_chat_id.set("qq:999:group")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


class TestRegistration:
    def test_tools_registered(self):
        from junjun_skills.plugins.pixiv import tools as pkg
        names = [t.name for t in pkg.TOOLS]
        assert names == ["pixiv_search_illusts", "pixiv_send_illust",
                         "pixiv_search_novels", "pixiv_download_novel"]

    def test_topic_keywords_pinned(self):
        from junjun_skills.registry import _TOPIC_KEYWORDS
        for name in ("pixiv_search_illusts", "pixiv_send_illust",
                     "pixiv_search_novels", "pixiv_download_novel"):
            assert name in _TOPIC_KEYWORDS and _TOPIC_KEYWORDS[name]


class TestSearchIllusts:
    @pytest.mark.asyncio
    async def test_recommend_list_with_links(self, monkeypatch):
        async def _search(keyword):
            assert keyword == "原神"
            return [{"kind": "illust", "id": "111", "title": "好图",
                     "author": "画师甲", "pages": 3},
                    {"kind": "illust", "id": "222", "title": "也好图",
                     "author": "画师乙", "pages": 1}]

        monkeypatch.setattr(illust, "_search_illusts", _search)
        _group()  # 群聊也能推荐
        out = await at.pixiv_search_illusts.ainvoke({"keyword": "原神"})
        assert "好图" in out and "画师甲" in out and "3页" in out
        assert "artworks/111" in out and "artworks/222" in out

    @pytest.mark.asyncio
    async def test_empty_result(self, monkeypatch):
        async def _search(keyword):
            return []

        monkeypatch.setattr(illust, "_search_illusts", _search)
        out = await at.pixiv_search_illusts.ainvoke({"keyword": "不存在"})
        assert "没找到" in out

    @pytest.mark.asyncio
    async def test_no_cookie(self, monkeypatch):
        monkeypatch.setattr(at, "_cookie", lambda: "")
        out = await at.pixiv_search_illusts.ainvoke({"keyword": "x"})
        assert "Cookie" in out


class TestSearchNovels:
    def _novel_result(self):
        return {"novel": {"data": [
            {"id": 61, "title": "某单篇", "userName": "作者甲", "xRestrict": 0},
            {"id": 62, "title": "某R18", "userName": "作者乙", "xRestrict": 2},
            {"id": 63, "title": "第3章", "userName": "作者丙", "xRestrict": 0,
             "seriesId": 555, "seriesTitle": "某系列"},
        ]}}

    @pytest.mark.asyncio
    async def test_recommend_filters_r18_and_links(self, monkeypatch):
        async def _search(keyword, page=1):
            return self._novel_result()

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        out = await at.pixiv_search_novels.ainvoke({"keyword": "原神"})
        assert "某单篇" in out and "show.php?id=61" in out
        assert "某R18" not in out  # R18 不进推荐清单
        assert "某系列" in out and "（系列）" in out and "series/555" in out

    @pytest.mark.asyncio
    async def test_error_result(self, monkeypatch):
        async def _search(keyword, page=1):
            return {"error": "HTTP 403"}

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        out = await at.pixiv_search_novels.ainvoke({"keyword": "x"})
        assert "失败" in out


class TestSendIllust:
    def _stubs(self, monkeypatch, xrestrict=0, pages=2):
        async def _detail(iid):
            return {"title": "某作品", "userName": "某画师",
                    "xRestrict": xrestrict, "pageCount": pages}

        async def _urls(iid, page_count):
            return ["https://i.pixiv.re/a_p0.jpg", "https://i.pixiv.re/a_p1.jpg"]

        monkeypatch.setattr(illust, "_illust_detail", _detail)
        monkeypatch.setattr(illust, "_illust_page_urls", _urls)

    @pytest.mark.asyncio
    async def test_group_rejected_with_guidance(self, monkeypatch):
        _group()
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "111"})
        assert "群聊" in out and "pixiv_search_illusts" in out  # 解释+指替代方案

    @pytest.mark.asyncio
    async def test_private_sends_images(self, _fake_gateway, monkeypatch):
        self._stubs(monkeypatch)
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "111"})
        assert "已把" in out and "某作品" in out
        rs = _fake_gateway[0]
        assert rs.target_user_id == "12345" and rs.target_group_id is None
        assert [s.type for s in rs.segments].count("image") == 2
        assert any("某画师" in (s.data or "") for s in rs.segments if s.type == "text")

    @pytest.mark.asyncio
    async def test_r18_rejected(self, monkeypatch):
        self._stubs(monkeypatch, xrestrict=2)
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "111"})
        assert "R18" in out

    @pytest.mark.asyncio
    async def test_bad_id(self):
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "abc"})
        assert "ID" in out


class TestDownloadNovel:
    @pytest.mark.asyncio
    async def test_group_rejected_with_guidance(self):
        _group()
        out = await at.pixiv_download_novel.ainvoke({"target": "12345"})
        assert "私聊" in out and "pixiv_search_novels" in out

    @pytest.mark.asyncio
    async def test_bare_digits_route_single(self, monkeypatch):
        calls = []

        async def _single(ctx, nid):
            calls.append(("single", nid, ctx.meta.user_id))
            return "《某篇》抓取完成，txt 已发你～"

        async def _series(ctx, sid):
            calls.append(("series", sid))
            return "开始抓取"

        monkeypatch.setattr(novel_mod, "_do_single", _single)
        monkeypatch.setattr(novel_mod, "_do_series", _series)
        out = await at.pixiv_download_novel.ainvoke({"target": "61"})
        assert calls == [("single", "61", "12345")]  # 裸数字 = 单篇（不是系列）
        assert "抓取完成" in out

    @pytest.mark.asyncio
    async def test_series_prefix_routes_series(self, monkeypatch):
        calls = []

        async def _series(ctx, sid):
            calls.append(sid)
            return "开始抓取系列 555"

        monkeypatch.setattr(novel_mod, "_do_series", _series)
        out = await at.pixiv_download_novel.ainvoke({"target": "series 555"})
        assert calls == ["555"]
        assert "开始抓取" in out

    @pytest.mark.asyncio
    async def test_series_url_routes_series(self, monkeypatch):
        calls = []

        async def _series(ctx, sid):
            calls.append(sid)
            return "开始抓取"

        monkeypatch.setattr(novel_mod, "_do_series", _series)
        await at.pixiv_download_novel.ainvoke(
            {"target": "https://www.pixiv.net/novel/series/777"})
        assert calls == ["777"]

    @pytest.mark.asyncio
    async def test_cooldown(self, monkeypatch):
        async def _single(ctx, nid):
            return "ok"

        monkeypatch.setattr(novel_mod, "_do_single", _single)
        at._dl_last["qq:12345:private"] = time.time()
        out = await at.pixiv_download_novel.ainvoke({"target": "61"})
        assert "秒后再试" in out
