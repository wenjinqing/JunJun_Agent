"""music 插件测试：搜索列表 / choose / 快捷数字选歌 / 限流 / play_music 工具。"""

import json
from types import SimpleNamespace

import pytest

from junjun_agent import commands, interceptors


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    interceptors.clear_interceptors()
    yield
    commands.clear_commands()
    interceptors.clear_interceptors()


@pytest.fixture(autouse=True)
def _clean_state():
    import junjun_skills.plugins.music.tools as music
    music._search_cache.clear()
    music._last_use.clear()
    yield
    music._search_cache.clear()
    music._last_use.clear()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text):
    return SimpleNamespace(text=text, user_id="12345", nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


def _songs(n=3, source="netease"):
    """造 n 条标准化搜索结果。"""
    return [{"source": source, "source_name": "网易云音乐", "id": str(i),
             "song": f"歌{i}", "singer": "歌手甲", "album": "专辑乙",
             "cover": "http://x/c.jpg", "url": f"http://x/{i}.mp3",
             "link": f"http://x/p{i}", "interval": "03:30"}
            for i in range(1, n + 1)]


def _patch_search(monkeypatch, music, fail_sources=()):
    """替换搜索 helper：fail_sources 里的源返回 None，其余返回 3 首歌。返回调用记录。"""
    calls = []

    async def _search(source, keyword, num=10):
        calls.append(source)
        return None if source in fail_sources else _songs(source=source)

    monkeypatch.setattr(music, "fetch_search", _search)
    return calls


def _patch_detail(monkeypatch, music, audio="http://x/1.mp3"):
    """替换详情 helper：返回第 1 首（可控制是否有音频直链）。"""

    async def _detail(source, keyword, index):
        d = _songs(source=source)[0]
        d["url"] = audio
        return d

    monkeypatch.setattr(music, "fetch_detail", _detail)


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_sends_list_and_caches(self, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music)
        result = await music.music_cmd(_ctx("/music 晴天"))
        assert "1. 歌1 - 歌手甲" in result and "3. 歌3 - 歌手甲" in result
        assert "choose" in result
        cache = music._search_cache["qq:999:group"]
        assert cache["keyword"] == "晴天" and len(cache["results"]) == 3

    @pytest.mark.asyncio
    async def test_fallback_netease_to_qq(self, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        calls = _patch_search(monkeypatch, music, fail_sources=("netease",))
        result = await music.music_cmd(_ctx("/music 晴天"))
        assert calls == ["netease", "qq"]  # 网易失败降级到 QQ
        assert music._search_cache["qq:999:group"]["source"] == "qq"
        assert "QQ" in result or "歌1" in result

    @pytest.mark.asyncio
    async def test_explicit_source_no_fallback(self, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        calls = _patch_search(monkeypatch, music, fail_sources=("juhe",))
        result = await music.music_cmd(_ctx("/music juhe 起风了"))
        assert calls == ["juhe"]  # 指定源不再降级
        assert "没找到" in result

    @pytest.mark.asyncio
    async def test_usage_when_no_keyword(self):
        import junjun_skills.plugins.music.tools as music
        result = await music.music_cmd(_ctx("/music"))
        assert "用法" in result

    @pytest.mark.asyncio
    async def test_rate_limit(self, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music)
        await music.music_cmd(_ctx("/music 晴天"))
        result = await music.music_cmd(_ctx("/music 稻香"))
        assert "秒" in result


class TestChoose:
    @pytest.mark.asyncio
    async def test_choose_valid_sends_music_card(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music)
        _patch_detail(monkeypatch, music)
        await music.music_cmd(_ctx("/music 晴天"))
        result = await music.choose_cmd(_ctx("/choose 1"))
        assert result is None
        segs = _fake_gateway[0].segments
        assert segs[0].type == "music"
        card = json.loads(segs[0].data)  # music 段 data 必须是可解析 JSON
        assert card["audio"] == "http://x/1.mp3"
        assert card["title"] == "歌1" and card["content"] == "歌手甲"
        assert segs[1].type == "text" and "歌1" in segs[1].data

    @pytest.mark.asyncio
    async def test_choose_no_cache(self):
        import junjun_skills.plugins.music.tools as music
        result = await music.choose_cmd(_ctx("/choose 1"))
        assert "搜索" in result

    @pytest.mark.asyncio
    async def test_choose_out_of_range(self, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music)
        await music.music_cmd(_ctx("/music 晴天"))
        result = await music.choose_cmd(_ctx("/choose 9"))
        assert "超出范围" in result

    @pytest.mark.asyncio
    async def test_no_audio_degrades_to_text(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music)
        _patch_detail(monkeypatch, music, audio="")  # 无音频直链
        await music.music_cmd(_ctx("/music 晴天"))
        await music.choose_cmd(_ctx("/choose 1"))
        segs = _fake_gateway[0].segments
        assert len(segs) == 1 and segs[0].type == "text"
        assert "失效" in segs[0].data


class TestQuickChoose:
    @pytest.mark.asyncio
    async def test_quick_choose_with_cache(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music)
        _patch_detail(monkeypatch, music)
        await music.music_cmd(_ctx("/music 晴天"))
        ctx = _ctx("2")
        ctx.args = "2"
        consumed = await music.quick_choose(ctx)
        assert consumed is True
        card = json.loads(_fake_gateway[0].segments[0].data)
        assert card["audio"] == "http://x/1.mp3"

    @pytest.mark.asyncio
    async def test_quick_choose_without_cache_passes(self, _fake_gateway):
        import junjun_skills.plugins.music.tools as music
        ctx = _ctx("1")
        ctx.args = "1"
        consumed = await music.quick_choose(ctx)
        assert consumed is False  # 无缓存：放行给正常决策
        assert not _fake_gateway


class TestPlayMusicTool:
    @pytest.mark.asyncio
    async def test_tool_sends_card_via_gateway(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        from junjun_skills.builtin import memory_skills
        _patch_search(monkeypatch, music)
        _patch_detail(monkeypatch, music)
        token = memory_skills.current_chat_id.set("qq:999:group")
        try:
            out = await music.play_music.ainvoke({"song_name": "晴天"})
        finally:
            memory_skills.current_chat_id.reset(token)
        assert "播放" in out and "歌1" in out
        rs = _fake_gateway[0]
        assert rs.platform == "qq" and rs.target_group_id == "999"
        card = json.loads(rs.segments[0].data)
        assert card["audio"] == "http://x/1.mp3"

    @pytest.mark.asyncio
    async def test_tool_not_found(self, monkeypatch):
        import junjun_skills.plugins.music.tools as music
        _patch_search(monkeypatch, music, fail_sources=("netease", "qq"))
        out = await music.play_music.ainvoke({"song_name": "不存在的歌"})
        assert "没找到" in out
