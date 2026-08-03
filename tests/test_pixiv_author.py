"""pixiv_novel 作者作品列表（/novel author）测试。"""

import time

import pytest

from junjun_skills.plugins.pixiv import novel as tools


def _profile_payload():
    """profile/all：2 个系列 + 3 个小说 ID（其中 1 个属于系列）。"""
    return {
        "novelSeries": [
            {"id": "16238650", "title": "系列甲", "userName": "猫猫",
             "xRestrict": 0, "displaySeriesContentCount": 5},
            {"id": "14998441", "title": "系列乙", "userName": "猫猫",
             "xRestrict": 2, "displaySeriesContentCount": 3},
        ],
        "novels": {"28665187": None, "28352452": None, "28104566": None},
    }


def _works_payload():
    """profile/novels：28665187 属于系列（应被过滤），其余为单篇。"""
    return {
        "works": {
            "28665187": {"id": "28665187", "title": "系列甲第一章", "userName": "猫猫",
                         "xRestrict": 0, "seriesId": "16238650", "seriesTitle": "系列甲"},
            "28352452": {"id": "28352452", "title": "单篇A", "userName": "猫猫",
                         "xRestrict": 2, "seriesId": None},
            "28104566": {"id": "28104566", "title": "单篇B", "userName": "猫猫",
                         "xRestrict": 0, "seriesId": None},
        }
    }


def _mock_fetch(monkeypatch):
    async def _fetch(url, referer=""):
        if "profile/all" in url:
            return _profile_payload()
        if "profile/novels" in url:
            return _works_payload()
        return {"error": "unexpected url"}

    monkeypatch.setattr(tools, "_fetch_json", _fetch)


class TestExtractUserId:
    def test_user_url(self):
        assert tools._extract_user_id("https://www.pixiv.net/users/16689973") == "16689973"

    def test_user_url_with_lang(self):
        assert tools._extract_user_id("https://www.pixiv.net/users/16689973/novels") == "16689973"

    def test_bare_digits(self):
        assert tools._extract_user_id("16689973") == "16689973"

    def test_garbage(self):
        assert tools._extract_user_id("猫猫酱") == ""


class TestFetchAuthorWorks:
    @pytest.mark.asyncio
    async def test_series_and_standalone_split(self, monkeypatch):
        """系列内章节被过滤，单篇按最新排序。"""
        _mock_fetch(monkeypatch)
        works = await tools._fetch_author_works("16689973")
        assert works["author"] == "猫猫"
        assert [s["title"] for s in works["series"]] == ["系列甲", "系列乙"]
        assert [n["id"] for n in works["novels"]] == ["28352452", "28104566"]
        assert works["novels"][0]["r18"] is True
        assert works["series"][1]["r18"] is True

    @pytest.mark.asyncio
    async def test_profile_error_propagates(self, monkeypatch):
        async def _fetch(url, referer=""):
            return {"error": "HTTP 403"}

        monkeypatch.setattr(tools, "_fetch_json", _fetch)
        works = await tools._fetch_author_works("1")
        assert works["error"] == "HTTP 403"


class TestDoAuthor:
    @pytest.mark.asyncio
    async def test_list_and_cache(self, monkeypatch):
        _mock_fetch(monkeypatch)
        tools._search_cache.clear()
        out = await tools._do_author("https://www.pixiv.net/users/16689973", "u1")
        assert "猫猫" in out
        assert "系列甲" in out and "单篇A" in out
        assert "系列甲第一章" not in out  # 系列内章节不单列
        assert "/novel dl" in out
        cached = tools._search_cache["u1"]
        assert len(cached["items"]) == 4  # 2 系列 + 2 单篇
        assert cached["items"][0]["series_id"] == "16238650"  # 系列在前
        assert cached["items"][2]["id"] == "28352452"

    @pytest.mark.asyncio
    async def test_dl_reuses_author_cache(self, monkeypatch):
        """作者列表缓存与 /novel dl 编号下载打通。"""
        _mock_fetch(monkeypatch)
        tools._search_cache.clear()
        await tools._do_author("16689973", "u1")

        called = {}

        class _Ctx:
            class meta:
                user_id = "u1"

        async def _series(ctx, sid):
            called["series"] = sid
            return "ok-series"

        async def _single(ctx, nid):
            called["single"] = nid
            return "ok-single"

        monkeypatch.setattr(tools, "_do_series", _series)
        monkeypatch.setattr(tools, "_do_single", _single)
        assert await tools._do_download_by_number(_Ctx(), "1") == "ok-series"
        assert called["series"] == "16238650"
        assert await tools._do_download_by_number(_Ctx(), "3") == "ok-single"
        assert called["single"] == "28352452"

    @pytest.mark.asyncio
    async def test_bad_uid(self):
        out = await tools._do_author("猫猫酱", "u1")
        assert "没识别到" in out

    @pytest.mark.asyncio
    async def test_no_works(self, monkeypatch):
        async def _fetch(url, referer=""):
            return {"novelSeries": [], "novels": {}}

        monkeypatch.setattr(tools, "_fetch_json", _fetch)
        out = await tools._do_author("999", "u1")
        assert "还没有公开的小说" in out
