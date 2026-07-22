"""pixiv_novel 插件测试：白名单/群聊/冷却、搜索缓存、dl、系列合成 txt、无 Cookie 降级。"""

import re
from types import SimpleNamespace

import pytest

import junjun_skills.plugins.pixiv_novel.tools as tools
from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


@pytest.fixture(autouse=True)
def _reset_state():
    """每个测试清空冷却与搜索缓存，避免跨测试污染。"""
    tools._last_use.clear()
    tools._search_cache.clear()
    yield
    tools._last_use.clear()
    tools._search_cache.clear()


def _session(is_group=False):
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


def _ctx(text, user_id="12345", is_group=False):
    return commands.CommandContext(
        session=_session(is_group), meta=_meta(text, user_id),
        args=text.split(" ", 1)[1] if " " in text else "")


@pytest.fixture
def _env(monkeypatch, tmp_path):
    """插件内部 config.toml 落到 tmp_path（白名单 12345），并配置假 Cookie。"""
    save_dir = tmp_path / "save"
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[auth]\nallow_qq_list = ["12345"]\n\n'
        "[features]\n"
        "cooldown_seconds = 60\n"
        "api_timeout = 30\n"
        "max_chapters_per_series = 50\n"
        f'save_dir = "{save_dir.as_posix()}"\n\n'
        '[network]\nproxy = ""\n',
        encoding="utf-8")
    monkeypatch.setattr(tools, "_CONFIG_PATH", cfg)
    monkeypatch.setenv("PIXIV_COOKIE", "PHPSESSID=111_abcd")
    monkeypatch.setattr(tools, "_CHAPTER_DELAY", 0)
    return save_dir


@pytest.fixture
def _fake_pixiv(monkeypatch):
    """假 Pixiv AJAX：搜索 2 条（单篇+系列）、系列 900 两章、单篇正文。"""
    calls = []

    async def _fetch(url, referer=""):
        calls.append(url)
        if "/ajax/search/novels/" in url:
            return {"novel": {"data": [
                {"id": "201", "title": "搜索结果单篇", "userName": "作者乙", "xRestrict": 0},
                {"id": "301", "seriesId": "900", "seriesTitle": "搜索系列",
                 "title": "第5话", "userName": "作者丙", "xRestrict": 1},
            ]}}
        if "/ajax/novel/series_content/" in url:
            return {"page": {"seriesContents": [
                {"id": "101", "title": "第一章 开始"},
                {"id": "102", "title": "第二章 继续"},
            ]}}
        if "/ajax/novel/series/" in url:
            return {"title": "测试系列", "userName": "作者甲", "total": 2}
        m = re.search(r"/ajax/novel/(\d+)", url)
        if m:
            nid = m.group(1)
            title = {"101": "第一章 开始", "102": "第二章 继续"}.get(nid, f"单篇{nid}")
            return {"title": title, "userName": "作者甲", "content": f"正文{nid}"}
        return {"error": "unknown url"}

    monkeypatch.setattr(tools, "_fetch_json", _fetch)
    return calls


@pytest.fixture
def _fake_upload(monkeypatch):
    uploads = []

    async def _upload(user_id, file_path, name=""):
        uploads.append({"user_id": user_id, "file_path": file_path, "name": name})
        return True

    monkeypatch.setattr("junjun_core.napcat_client.upload_private_file", _upload)
    return uploads


class TestAccess:
    @pytest.mark.asyncio
    async def test_whitelist_pass(self, _env, _fake_pixiv, _fake_gateway):
        result = await tools.novel_cmd(_ctx("/novel search 测试"))
        assert "搜索结果单篇" in result
        assert "搜索系列" in result

    @pytest.mark.asyncio
    async def test_non_whitelist_rejected(self, _env, _fake_pixiv, _fake_gateway):
        result = await tools.novel_cmd(_ctx("/novel search 测试", user_id="99999"))
        assert "没有对你开放" in result
        assert _fake_pixiv == []          # 未触发任何网络请求
        assert _fake_gateway == []        # 友好文本走返回值，不走 security 上报

    @pytest.mark.asyncio
    async def test_group_rejected(self, _env, _fake_pixiv, _fake_gateway):
        result = await tools.novel_cmd(_ctx("/novel 900", is_group=True))
        assert "私聊" in result
        assert _fake_pixiv == []


class TestSearchAndDl:
    @pytest.mark.asyncio
    async def test_search_list_and_cache(self, _env, _fake_pixiv, _fake_gateway):
        result = await tools.novel_cmd(_ctx("/novel search 测试"))
        assert "1. 搜索结果单篇 [单篇]" in result
        assert "2. 搜索系列 [R18] [系列 900]" in result
        assert "/novel dl" in result
        # 缓存 10 分钟，键 user_id
        entry = tools._search_cache.get("12345")
        assert entry and len(entry["items"]) == 2

    @pytest.mark.asyncio
    async def test_dl_valid_single(self, _env, _fake_pixiv, _fake_gateway, _fake_upload):
        await tools.novel_cmd(_ctx("/novel search 测试"))
        tools._last_use.clear()            # 跳过冷却，专注验证 dl
        result = await tools.novel_cmd(_ctx("/novel dl 1"))
        assert "抓取完成" in result
        assert len(_fake_upload) == 1
        up = _fake_upload[0]
        assert up["user_id"] == "12345"
        assert up["name"].endswith(".txt") and "201" in up["name"]
        content = open(up["file_path"], encoding="utf-8").read()
        assert "正文201" in content and "单篇201" in content

    @pytest.mark.asyncio
    async def test_dl_valid_series(self, _env, _fake_pixiv, _fake_gateway, _fake_upload):
        await tools.novel_cmd(_ctx("/novel search 测试"))
        tools._last_use.clear()
        result = await tools.novel_cmd(_ctx("/novel dl 2"))
        assert "抓取完成" in result
        assert len(_fake_upload) == 1
        content = open(_fake_upload[0]["file_path"], encoding="utf-8").read()
        assert "测试系列" in content
        assert content.index("第一章 开始") < content.index("第二章 继续")
        assert "正文101" in content and "正文102" in content

    @pytest.mark.asyncio
    async def test_dl_invalid(self, _env, _fake_pixiv, _fake_gateway):
        # 无搜索记录
        assert "没有搜索记录" in await tools.novel_cmd(_ctx("/novel dl 1"))
        tools._last_use.clear()
        await tools.novel_cmd(_ctx("/novel search 测试"))
        tools._last_use.clear()
        assert "超出范围" in await tools.novel_cmd(_ctx("/novel dl 9"))
        tools._last_use.clear()
        assert "有效的编号" in await tools.novel_cmd(_ctx("/novel dl abc"))


class TestSeriesAndRead:
    @pytest.mark.asyncio
    async def test_series_compose_txt(self, _env, _fake_pixiv, _fake_gateway, _fake_upload):
        result = await tools.novel_cmd(
            _ctx("/novel https://www.pixiv.net/novel/series/900"))
        assert "抓取完成" in result and "2/2" in result
        # 进度提示先行（开始抓取/共2章）
        texts = [s.data for rs in _fake_gateway for s in rs.segments if s.type == "text"]
        assert any("开始抓取" in t for t in texts)
        assert any("共 2 章" in t for t in texts)
        # 发文件调用
        assert len(_fake_upload) == 1
        up = _fake_upload[0]
        assert up["user_id"] == "12345" and "900" in up["name"]
        # txt 内容与章节顺序
        content = open(up["file_path"], encoding="utf-8").read()
        assert "测试系列" in content and "作者甲" in content
        assert content.index("第一章 开始") < content.index("第二章 继续")
        assert "正文101" in content and "正文102" in content

    @pytest.mark.asyncio
    async def test_series_by_id_and_saved_under_save_dir(self, _env, _fake_pixiv,
                                                         _fake_gateway, _fake_upload):
        await tools.novel_cmd(_ctx("/novel 900"))
        assert len(_fake_upload) == 1
        path = _fake_upload[0]["file_path"]
        assert path.startswith(str(_env)) or str(_env) in path
        assert "测试系列_900.txt" in path

    @pytest.mark.asyncio
    async def test_read_single(self, _env, _fake_pixiv, _fake_gateway, _fake_upload):
        result = await tools.novel_cmd(_ctx("/novel read 201"))
        assert "抓取完成" in result
        assert len(_fake_upload) == 1
        content = open(_fake_upload[0]["file_path"], encoding="utf-8").read()
        assert "正文201" in content

    @pytest.mark.asyncio
    async def test_read_rejects_series_url(self, _env, _fake_pixiv, _fake_gateway):
        result = await tools.novel_cmd(
            _ctx("/novel read https://www.pixiv.net/novel/series/900"))
        assert "单篇" in result

    @pytest.mark.asyncio
    async def test_list_toc(self, _env, _fake_pixiv, _fake_gateway):
        result = await tools.novel_cmd(_ctx("/novel list 900"))
        assert "测试系列" in result and "共 2 章" in result
        assert result.index("第一章 开始") < result.index("第二章 继续")
        assert "id:101" in result and "id:102" in result
        # list 不发文件不抓正文（仅 meta + 目录两次请求）
        assert not any(re.search(r"/ajax/novel/\d+", u) for u in _fake_pixiv)


class TestCooldownAndCookie:
    @pytest.mark.asyncio
    async def test_cooldown(self, _env, _fake_pixiv, _fake_gateway):
        first = await tools.novel_cmd(_ctx("/novel list 900"))
        assert "共 2 章" in first
        second = await tools.novel_cmd(_ctx("/novel list 900"))
        assert "冷却" in second
        # 同一用户其他子命令同样被冷却拦截
        third = await tools.novel_cmd(_ctx("/novel search 测试"))
        assert "冷却" in third

    @pytest.mark.asyncio
    async def test_no_cookie_degrade(self, _env, _fake_pixiv, _fake_gateway, monkeypatch):
        monkeypatch.delenv("PIXIV_COOKIE", raising=False)
        result = await tools.novel_cmd(_ctx("/novel search 测试"))
        assert "不可用" in result and "PIXIV_COOKIE" in result
        assert _fake_pixiv == []
        # 无 Cookie 不消耗冷却：配上 Cookie 后可立即用
        monkeypatch.setenv("PIXIV_COOKIE", "PHPSESSID=111_abcd")
        ok = await tools.novel_cmd(_ctx("/novel search 测试"))
        assert "搜索结果单篇" in ok
