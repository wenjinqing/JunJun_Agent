"""pixiv 插件 /pixiv 命令族测试（2026-08-03 新端点，全部官方 API）：

search（搜索列表+R18 过滤）、rank（排行榜顶层 JSON）、new（关注新图）、
author（插画+小说合并）、dl（插画发图/小说私聊限定）、冷却。
"""

from types import SimpleNamespace

import pytest

import junjun_skills.plugins.pixiv.illust as illust
import junjun_skills.plugins.pixiv.novel as novel_mod
from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


@pytest.fixture(autouse=True)
def _reset_state():
    illust._last_use.clear()
    illust._list_cache.clear()
    yield
    illust._last_use.clear()
    illust._list_cache.clear()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group,
                           chat_id="qq:999:group" if is_group else "qq:12345:private")


def _meta(text, user_id="12345"):
    return SimpleNamespace(text=text, user_id=user_id, nickname="甲",
                           at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


@pytest.fixture(autouse=True)
def _cookie(monkeypatch):
    monkeypatch.setattr(illust, "_cookie", lambda: "PHPSESSID=1_x")
    monkeypatch.setattr(novel_mod, "_cookie", lambda: "PHPSESSID=1_x")


def _ctx(text, is_group=True, user_id="12345"):
    return commands.CommandContext(
        session=_session(is_group), meta=_meta(text, user_id),
        args=text.split(" ", 1)[1] if " " in text else "")


def _illust_item(iid, title="某图", r18=0, author="某画师"):
    return {"id": str(iid), "title": title, "userName": author,
            "xRestrict": r18, "aiType": 1, "pageCount": 1,
            "width": 1000, "height": 1200}


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_list_and_r18_filtered(self, monkeypatch):
        async def _fetch(url, referer=""):
            assert "mode=safe" in url
            return {"illustManga": {"data": [
                _illust_item(1, "好图"),
                _illust_item(2, "R18图", r18=2),  # 必须被滤掉
                _illust_item(3, "也好图"),
            ]}}

        # 搜索走 client.search_artworks（收藏分层池），patch client 层
        import junjun_skills.plugins.pixiv.client as pixiv_client
        monkeypatch.setattr(pixiv_client, "_fetch_json", _fetch)
        out = await illust.pixiv_cmd(_ctx("/pixiv search 原神"))
        assert "好图" in out and "也好图" in out
        assert "R18图" not in out
        assert "dl" in out

    @pytest.mark.asyncio
    async def test_search_empty(self, monkeypatch):
        async def _fetch(url, referer=""):
            return {"illustManga": {"data": []}}

        import junjun_skills.plugins.pixiv.client as pixiv_client
        monkeypatch.setattr(pixiv_client, "_fetch_json", _fetch)
        out = await illust.pixiv_cmd(_ctx("/pixiv search 不存在的东西"))
        assert "没找到" in out


class TestRank:
    @pytest.mark.asyncio
    async def test_rank_daily_illust(self, monkeypatch):
        async def _raw(url, referer=""):
            assert "mode=daily" in url and "content=illust" in url
            return {"contents": [
                {"rank": 1, "illust_id": 11, "title": "榜首", "user_name": "大佬",
                 "illust_content_type": {"sexual": 0}, "illust_page_count": "1",
                 "width": 1000, "height": 800},
                {"rank": 2, "illust_id": 12, "title": "R18榜眼", "user_name": "x",
                 "illust_content_type": {"sexual": 2}, "illust_page_count": "1",
                 "width": 1000, "height": 800},
            ]}

        monkeypatch.setattr(illust, "_fetch_raw", _raw)
        out = await illust.pixiv_cmd(_ctx("/pixiv rank"))
        assert "每日插画榜" in out and "榜首" in out and "第1名" in out
        assert "R18榜眼" not in out

    @pytest.mark.asyncio
    async def test_rank_manga_weekly(self, monkeypatch):
        async def _raw(url, referer=""):
            assert "mode=weekly" in url and "content=manga" in url
            return {"contents": [
                {"rank": 1, "illust_id": 21, "title": "漫画王", "user_name": "y",
                 "illust_content_type": {"sexual": 0}, "illust_page_count": "4",
                 "width": 1000, "height": 800},
            ]}

        monkeypatch.setattr(illust, "_fetch_raw", _raw)
        out = await illust.pixiv_cmd(_ctx("/pixiv rank weekly manga"))
        assert "每周漫画榜" in out and "漫画王" in out


class TestNew:
    @pytest.mark.asyncio
    async def test_follow_latest(self, monkeypatch):
        async def _fetch(url, referer=""):
            assert "follow_latest" in url
            return {"page": {"ids": [1]}, "thumbnails": {"illust": [
                _illust_item(31, "新图"), _illust_item(32, "AI新图"),
            ]}}

        monkeypatch.setattr(illust, "_fetch_json", _fetch)
        out = await illust.pixiv_cmd(_ctx("/pixiv new"))
        assert "关注画师的新图" in out and "新图" in out


class TestAuthor:
    @pytest.mark.asyncio
    async def test_author_merges_illust_and_novel(self, monkeypatch):
        async def _illust_fetch(url, referer=""):
            if "profile/all" in url:
                return {"illusts": {"41": {}, "42": {}}}
            if "profile/illusts" in url:
                return {"works": {
                    "41": {"title": "画师图1", "userName": "某作者",
                           "xRestrict": 0, "pageCount": 1},
                    "42": {"title": "画师图2", "userName": "某作者",
                           "xRestrict": 0, "pageCount": 1},
                }}
            return {"error": "unexpected"}

        async def _novel_fetch(url, referer=""):
            if "profile/all" in url:
                return {"novelSeries": [{"id": 555, "title": "某系列",
                                         "userName": "某作者", "xRestrict": 0,
                                         "displaySeriesContentCount": 12}],
                        "novels": {"61": {}}}
            return {"works": {"61": {"title": "某单篇", "userName": "某作者",
                                     "xRestrict": 0}}}

        monkeypatch.setattr(illust, "_fetch_json", _illust_fetch)
        monkeypatch.setattr(novel_mod, "_fetch_json", _novel_fetch)
        out = await illust.pixiv_cmd(_ctx("/pixiv author 16689973"))
        assert "画师图1" in out and "某系列" in out and "某单篇" in out
        assert "插画" in out and "小说系列" in out


class TestDl:
    @pytest.mark.asyncio
    async def test_dl_illust_sends_images(self, _fake_gateway, monkeypatch):
        async def _fetch(url, referer=""):
            if "/pages" in url:
                return [{"urls": {"regular": "https://i.pximg.net/a_p0.jpg"}},
                        {"urls": {"regular": "https://i.pximg.net/a_p1.jpg"}}]
            return {"title": "某作品", "userName": "某画师", "xRestrict": 0,
                    "pageCount": 2, "urls": {"regular": "https://i.pximg.net/a_p0.jpg"}}

        monkeypatch.setattr(illust, "_fetch_json", _fetch)

        async def _b64(urls):
            return [f"base64://fake-{u}" for u in urls]  # 本侧代下（NapCat 拉不到图床）

        monkeypatch.setattr(illust, "images_to_b64", _b64)
        illust._list_cache["12345"] = {
            "ts": __import__("time").time(),
            "items": [{"kind": "illust", "id": "147971647", "title": "某作品",
                       "author": "某画师", "pages": 2}]}
        out = await illust.pixiv_cmd(_ctx("/pixiv dl 1"))
        assert out is None  # 已发送
        segs = _fake_gateway[0].segments
        assert [s.type for s in segs].count("image") == 2
        # 代下转 base64：NapCat 直连图床超时（2026-08-03 实锤），不再发 URL
        assert all(s.data.startswith("base64://") for s in segs if s.type == "image")
        assert any("某作品" in (s.data or "") for s in segs if s.type == "text")

    @pytest.mark.asyncio
    async def test_dl_novel_rejected_in_group(self, monkeypatch):
        import time
        illust._list_cache["12345"] = {
            "ts": time.time(),
            "items": [{"kind": "novel", "id": "61", "title": "某小说",
                       "author": "某作者", "pages": "单篇"}]}
        out = await illust.pixiv_cmd(_ctx("/pixiv dl 1", is_group=True))
        assert "私聊" in out

    @pytest.mark.asyncio
    async def test_dl_expired_cache(self):
        out = await illust.pixiv_cmd(_ctx("/pixiv dl 1"))
        assert "过期" in out or "重新" in out


class TestIllustDirect:
    @pytest.mark.asyncio
    async def test_illust_url_and_multipage_cap(self, _fake_gateway, monkeypatch):
        async def _fetch(url, referer=""):
            if "/pages" in url:
                return [{"urls": {"regular": f"https://i.pximg.net/a_p{i}.jpg"}}
                        for i in range(5)]  # 5 页，应只发 3
            return {"title": "多页作品", "userName": "某画师", "xRestrict": 0,
                    "pageCount": 5}

        monkeypatch.setattr(illust, "_fetch_json", _fetch)

        async def _b64(urls):
            return [f"base64://fake-{u}" for u in urls]

        monkeypatch.setattr(illust, "images_to_b64", _b64)
        out = await illust.pixiv_cmd(
            _ctx("/pixiv illust https://www.pixiv.net/artworks/147971647"))
        assert out is None
        segs = _fake_gateway[0].segments
        assert [s.type for s in segs].count("image") == 3
        assert any("共 5 页" in (s.data or "") for s in segs if s.type == "text")

    @pytest.mark.asyncio
    async def test_illust_r18_rejected(self, monkeypatch):
        async def _fetch(url, referer=""):
            return {"title": "大人图", "userName": "x", "xRestrict": 2, "pageCount": 1}

        monkeypatch.setattr(illust, "_fetch_json", _fetch)
        out = await illust.pixiv_cmd(_ctx("/pixiv illust 123"))
        assert "R18" in out


class TestImageB64:
    @pytest.mark.asyncio
    async def test_images_to_b64_skips_failures(self, monkeypatch):
        from junjun_skills.plugins.pixiv import client

        async def _one(url):
            return "base64://ok" if "good" in url else ""

        monkeypatch.setattr(client, "fetch_image_b64", _one)
        out = await client.images_to_b64(["http://x/good1", "http://x/bad",
                                          "http://x/good2"])
        assert out == ["base64://ok", "base64://ok"]

    @pytest.mark.asyncio
    async def test_fetch_image_b64_http_error(self, monkeypatch):
        from junjun_skills.plugins.pixiv import client

        class _Resp:
            status_code = 404
            content = b""

        class _Sess:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None, timeout=None):
                return _Resp()

        import curl_cffi.requests as cc
        monkeypatch.setattr(cc, "AsyncSession", lambda **kw: _Sess())
        assert await client.fetch_image_b64("http://x/missing.jpg") == ""


class TestSafetyFilter:
    """R18 三层元数据过滤（2026-08-03 用户实锤 mode=safe 漏擦边/R18）。"""

    def test_is_safe_item_matrix(self):
        from junjun_skills.plugins.pixiv.client import is_safe_item
        ok = {"xRestrict": 0, "sl": 2, "tags": ["原神"]}
        assert is_safe_item(ok, group=True) and is_safe_item(ok, group=False)
        # xRestrict 双保险
        assert not is_safe_item({"xRestrict": 2}, group=True)
        assert not is_safe_item({"xRestrict": 2}, group=False)
        # tag 黑名单（私聊也拦）
        assert not is_safe_item({"xRestrict": 0, "tags": ["R-18"]}, group=False)
        # 详情形态 tags（dict 包 list[dict]）
        detail = {"xRestrict": 0, "tags": {"tags": [{"tag": "NSFW"}]}}
        assert not is_safe_item(detail, group=False)
        # sl 擦边：群聊拦、私聊放
        border = {"xRestrict": 0, "sl": 4, "tags": []}
        assert not is_safe_item(border, group=True)
        assert is_safe_item(border, group=False)

    @pytest.mark.asyncio
    async def test_ranking_group_drops_sexual_1(self, monkeypatch):
        async def _raw(url, referer=""):
            return {"contents": [
                {"rank": 1, "illust_id": 11, "title": "健全", "user_name": "a",
                 "illust_content_type": {"sexual": 0}, "illust_page_count": "1"},
                {"rank": 2, "illust_id": 12, "title": "轻度擦边", "user_name": "b",
                 "illust_content_type": {"sexual": 1}, "illust_page_count": "1"},
                {"rank": 3, "illust_id": 13, "title": "露骨", "user_name": "c",
                 "illust_content_type": {"sexual": 2}, "illust_page_count": "1"},
            ]}

        monkeypatch.setattr(illust, "_fetch_raw", _raw)
        group_items = await illust._ranking("daily", "illust", group=True)
        assert [i["title"] for i in group_items] == ["健全"]  # 群聊只留 sexual==0
        priv_items = await illust._ranking("daily", "illust", group=False)
        assert [i["title"] for i in priv_items] == ["健全", "轻度擦边"]

    @pytest.mark.asyncio
    async def test_setu_bookmark_threshold(self, monkeypatch):
        """收藏门槛：低于 min_bookmarks 的图被详情层刷掉。"""
        from junjun_skills.plugins.pixiv import setu
        monkeypatch.setattr(setu, "_min_bookmarks", lambda: 300)

        async def _fetch(url, referer=""):
            return {"title": "冷门图", "userName": "新人", "xRestrict": 0,
                    "bookmarkCount": 12, "urls": {"regular": "http://x/a.jpg"}}

        monkeypatch.setattr(setu, "_fetch_json", _fetch)
        assert await setu._illust_image_urls("1", group=False) == ("", [])

        async def _fetch_hot(url, referer=""):
            return {"title": "热门图", "userName": "大佬", "xRestrict": 0,
                    "bookmarkCount": 5000, "urls": {"regular": "http://x/b.jpg"}}

        monkeypatch.setattr(setu, "_fetch_json", _fetch_hot)
        info, urls = await setu._illust_image_urls("2", group=False)
        assert "热门图" in info and urls

    @pytest.mark.asyncio
    async def test_setu_detail_r18_tag_dropped(self, monkeypatch):
        from junjun_skills.plugins.pixiv import setu
        monkeypatch.setattr(setu, "_min_bookmarks", lambda: 0)

        async def _fetch(url, referer=""):
            return {"title": "漏网", "userName": "x", "xRestrict": 0,
                    "bookmarkCount": 999, "tags": {"tags": [{"tag": "R-18"}]},
                    "urls": {"regular": "http://x/c.jpg"}}

        monkeypatch.setattr(setu, "_fetch_json", _fetch)
        assert await setu._illust_image_urls("3", group=False) == ("", [])


class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown(self, monkeypatch):
        async def _fetch(url, referer=""):
            return {"illustManga": {"data": [_illust_item(1)]}}

        monkeypatch.setattr(illust, "_fetch_json", _fetch)
        await illust.pixiv_cmd(_ctx("/pixiv search x"))
        out = await illust.pixiv_cmd(_ctx("/pixiv search x"))
        assert "冷却" in out
