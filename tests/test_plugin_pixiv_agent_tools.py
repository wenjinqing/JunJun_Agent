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


class TestAuthorFallback:
    """作者名反查（Premium 限定绕过）：作品同名作者 -> touch -> 搜索引擎。"""

    @pytest.mark.asyncio
    async def test_works_author_hit(self, monkeypatch):
        """第 1 级：作品搜索结果里有同名作者 -> 直接定位。"""
        async def _search(keyword, page=1):
            return {"novel": {"data": [
                {"userName": "fdkgf", "userId": "80626680", "title": "别人的"},
                {"userName": "爱丽丝猫猫酱", "userId": "16689973", "title": "她自己的"},
            ]}}

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        uid, name = await novel_mod._search_user_by_name("爱丽丝猫猫酱")
        assert uid == "16689973" and name == "爱丽丝猫猫酱"

    @pytest.mark.asyncio
    async def test_touch_hit(self, monkeypatch):
        async def _search(keyword, page=1):
            return {"novel": {"data": []}}  # 第 1 级不中

        async def _fetch(url, referer=""):
            assert "touch/ajax/search/users" in url
            return {"users": [{"user_id": "12345", "user_name": "爱丽丝猫猫酱"}],
                    "total": 1}

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        monkeypatch.setattr(novel_mod, "_fetch_json", _fetch)
        uid, name = await novel_mod._search_user_by_name("爱丽丝猫猫酱")
        assert uid == "12345" and name == "爱丽丝猫猫酱"

    @pytest.mark.asyncio
    async def test_google_fallback(self, monkeypatch):
        async def _search(keyword, page=1):
            return {"novel": {"data": []}}

        async def _fetch(url, referer=""):
            return {"users": [], "total": 0}  # touch 恒空（非会员）

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        monkeypatch.setattr(novel_mod, "_fetch_json", _fetch)

        async def _gsearch(query, num_results=10):
            assert "site:pixiv.net/users" in query
            return [{"url": "https://www.pixiv.net/users/99887",
                     "title": "爱丽丝猫猫酱"}]

        import junjun_skills.plugins.google_search.tools as gsearch
        monkeypatch.setattr(gsearch, "_search_with_fallback", _gsearch)
        uid, name = await novel_mod._search_user_by_name("爱丽丝猫猫酱")
        assert uid == "99887"

    @pytest.mark.asyncio
    async def test_not_found(self, monkeypatch):
        async def _search(keyword, page=1):
            return {"novel": {"data": []}}

        async def _fetch(url, referer=""):
            return {"users": [], "total": 0}

        async def _gsearch(query, num_results=10):
            return [{"url": "https://www.pixiv.net/novel/show.php?id=1",
                     "title": "无关"}]

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        monkeypatch.setattr(novel_mod, "_fetch_json", _fetch)
        import junjun_skills.plugins.google_search.tools as gsearch
        monkeypatch.setattr(gsearch, "_search_with_fallback", _gsearch)
        assert await novel_mod._search_user_by_name("不存在的人") == ("", "")

    @pytest.mark.asyncio
    async def test_cmd_search_falls_back_to_author(self, monkeypatch):
        async def _search(keyword, page=1):
            # 命令的 _do_search 调用为空（触发降级）；
            # 反查的第 1 级再次调用也空（逼到 touch 级）
            return {"novel": {"data": []}}

        async def _fetch(url, referer=""):
            return {"users": [{"user_id": "12345", "user_name": "猫猫"}]}

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        monkeypatch.setattr(novel_mod, "_fetch_json", _fetch)

        async def _author(uid, user_id):
            return f"作者「猫猫」的作品（系列 1 / 单篇 3）：..."

        monkeypatch.setattr(novel_mod, "_do_author", _author)
        out = await novel_mod._do_search("猫猫", "u1")
        assert "找到了作者「猫猫」" in out and "单篇 3" in out

    @pytest.mark.asyncio
    async def test_tool_author_recommend(self, monkeypatch):
        async def _search(keyword, page=1):
            return {"novel": {"data": []}}

        async def _fetch(url, referer=""):
            return {"users": [{"user_id": "12345", "user_name": "猫猫"}]}

        async def _works(uid):
            return {"author": "猫猫",
                    "series": [{"series_id": "555", "title": "某系列",
                                "author": "猫猫", "r18": False}],
                    "novels": [{"id": "61", "title": "某单篇", "author": "猫猫",
                                "r18": False},
                               {"id": "62", "title": "大人的", "author": "猫猫",
                                "r18": True}]}

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        monkeypatch.setattr(novel_mod, "_fetch_json", _fetch)
        monkeypatch.setattr(novel_mod, "_fetch_author_works", _works)
        out = await at.pixiv_search_novels.ainvoke({"keyword": "猫猫"})  # 私聊
        assert "找到了作者「猫猫」" in out
        assert "series/555" in out and "show.php?id=61" in out
        assert "【R18】" in out  # 私聊 R18 打标保留
        _group()
        monkeypatch.setattr(novel_mod, "_fetch_author_works", _works)
        out = await at.pixiv_search_novels.ainvoke({"keyword": "猫猫"})
        assert "大人的" not in out  # 群聊滤掉 R18


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
        seen = {}

        async def _search(keyword, group=True):
            seen["group"] = group
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
        assert seen["group"] is True  # 群聊场景透传严格过滤

    @pytest.mark.asyncio
    async def test_empty_result(self, monkeypatch):
        async def _search(keyword, group=True):
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
    async def test_recommend_group_filters_r18(self, monkeypatch):
        async def _search(keyword, page=1):
            return self._novel_result()

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        _group()
        out = await at.pixiv_search_novels.ainvoke({"keyword": "原神"})
        assert "某单篇" in out and "show.php?id=61" in out
        assert "某R18" not in out  # 群聊推荐清单不含 R18
        assert "某系列" in out and "（系列）" in out and "series/555" in out

    @pytest.mark.asyncio
    async def test_recommend_private_marks_r18(self, monkeypatch):
        async def _search(keyword, page=1):
            return self._novel_result()

        monkeypatch.setattr(novel_mod, "_search_novels", _search)
        out = await at.pixiv_search_novels.ainvoke({"keyword": "原神"})  # 默认私聊
        assert "【R18】" in out and "某R18" in out  # 私聊保留 R18 并打标

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

        async def _b64(urls):
            return [f"base64://fake-{u}" for u in urls]  # 本侧代下

        monkeypatch.setattr(at, "images_to_b64", _b64)
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "111"})
        assert "已把" in out and "某作品" in out
        rs = _fake_gateway[0]
        assert rs.target_user_id == "12345" and rs.target_group_id is None
        assert [s.type for s in rs.segments].count("image") == 2
        assert all(s.data.startswith("base64://")
                   for s in rs.segments if s.type == "image")
        assert any("某画师" in (s.data or "") for s in rs.segments if s.type == "text")

    @pytest.mark.asyncio
    async def test_r18_allowed_gore_rejected(self, monkeypatch):
        # 私聊放开 R18（xRestrict=1 放行）
        self._stubs(monkeypatch, xrestrict=1)

        async def _b64(urls):
            return ["base64://x"]

        monkeypatch.setattr(at, "images_to_b64", _b64)
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "111"})
        assert "已把" in out
        # R-18G（xRestrict=2）私聊也拒
        self._stubs(monkeypatch, xrestrict=2)
        out = await at.pixiv_send_illust.ainvoke({"illust_id": "111"})
        assert "R-18G" in out

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
