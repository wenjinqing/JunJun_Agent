"""抖音内容理解测试：SSR 解析/材料构建/缓存/直播婉拒/拦截器语义分流/最近清单注入。"""

import json
from types import SimpleNamespace

import pytest

import junjun_core.config.config as cfg_mod
from junjun_skills.plugins.douyin import content as dc
from junjun_skills.plugins.douyin import tools as dt


@pytest.fixture(autouse=True)
def env():
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"douyin": {"enable_understand": True}})
    dc._MATERIAL_CACHE.clear(); dc._PENDING.clear(); dc._RECENT.clear()
    yield
    dc._MATERIAL_CACHE.clear(); dc._PENDING.clear(); dc._RECENT.clear()
    cfg_mod.global_config = old


_ITEM = {
    "desc": "每日推荐一个MC整合包，今天推荐的是「终焉之诗」 #游戏精选 #minecraft",
    "aweme_id": "7669014018669006089",
    "author": {"nickname": "莱姆Lime"},
    "music": {"title": "莱姆Lime创作的原声"},
    "statistics": {"digg_count": 8883, "comment_count": 437, "share_count": 8134},
    "video": {"duration": 142034,
              "play_addr": {"url_list": ["https://aweme.snssdk.com/aweme/v1/playwm/?video_id=abc"]}},
}


def _router_html(item=None):
    data = {"loaderData": {"video_(id)/page": {
        "videoInfoRes": {"status_code": 0, "item_list": [item or _ITEM]}}}}
    return f'<script>window._ROUTER_DATA = {json.dumps(data, ensure_ascii=False)}</script>'


class TestParseRouterData:
    def test_parse_ok(self):
        item = dc._parse_router_data(_router_html())
        assert item["aweme_id"] == "7669014018669006089"

    def test_parse_no_marker(self):
        assert dc._parse_router_data("<html>verify page</html>") is None

    def test_parse_bad_json(self):
        assert dc._parse_router_data(
            "window._ROUTER_DATA = {oops</script>") is None

    def test_item_to_info(self):
        info = dc._item_to_info("7669014018669006089", _ITEM)
        assert info["author"] == "莱姆Lime"
        assert info["duration_s"] == 142
        assert info["digg"] == 8883
        assert info["play_url"].startswith("https://")


class _FakeResp:
    def __init__(self, text="", headers=None, url=""):
        self.text = text
        self.headers = headers or {}
        self.url = url


def _stub_client(mp, *, location="", share_html=""):
    """伪造 httpx.AsyncClient：第一次 GET 返回 302 location，第二次返回分享页。"""
    calls = {"n": 0}

    class _Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **kw):
            calls["n"] += 1
            if calls["n"] == 1 and location:
                return _FakeResp(headers={"location": location})
            return _FakeResp(text=share_html)

    import httpx
    mp.setattr(httpx, "AsyncClient", _Client)
    return calls


class TestResolve:
    @pytest.mark.asyncio
    async def test_short_link_full_chain(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://www.iesdouyin.com/share/video/7669014018669006089/?x=1",
                     share_html=_router_html())
        info = await dc._resolve_video("https://v.douyin.com/FHalxoVyefQ/")
        assert info["aweme_id"] == "7669014018669006089"
        assert info["author"] == "莱姆Lime"

    @pytest.mark.asyncio
    async def test_live_link_detected(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://webcast.amemv.com/douyin/webcast/reflow/123?u_code=x")
        info = await dc._resolve_video("https://v.douyin.com/zHg0ioiDMRw/")
        assert info == {"type": "live"}

    @pytest.mark.asyncio
    async def test_no_video_id_returns_none(self, monkeypatch):
        _stub_client(monkeypatch, location="https://www.douyin.com/user/MS4wLjAB")
        assert await dc._resolve_video("https://v.douyin.com/abc/") is None

    @pytest.mark.asyncio
    async def test_verify_page_returns_none(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://www.iesdouyin.com/share/video/1234567890123456789/",
                     share_html="<html>滑块验证</html>")
        assert await dc._resolve_video("https://v.douyin.com/abc/") is None


class TestMaterial:
    @pytest.mark.asyncio
    async def test_material_and_cache(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://www.iesdouyin.com/share/video/7669014018669006089/",
                     share_html=_router_html())
        m = await dc.get_material("https://v.douyin.com/FHalxoVyefQ/")
        assert m and "终焉之诗" in m["material"]
        assert "莱姆Lime" in m["material"] and "8883赞" in m["material"]
        # 缓存命中：客户端炸掉也能取到
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", None)
        m2 = await dc.get_material("https://v.douyin.com/FHalxoVyefQ/")
        assert m2 == m

    @pytest.mark.asyncio
    async def test_live_returns_live_dict(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://webcast.amemv.com/douyin/webcast/reflow/123")
        m = await dc.get_material("https://v.douyin.com/xyz/")
        assert m == {"type": "live"}

    @pytest.mark.asyncio
    async def test_play_url_from_cache(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://www.iesdouyin.com/share/video/7669014018669006089/",
                     share_html=_router_html())
        url = await dc.get_play_url("https://v.douyin.com/FHalxoVyefQ/")
        assert "playwm" in url


class TestRecentBlock:
    @pytest.mark.asyncio
    async def test_prewarm_populates_recent(self, monkeypatch):
        _stub_client(monkeypatch,
                     location="https://www.iesdouyin.com/share/video/7669014018669006089/",
                     share_html=_router_html())

        class _FakeModel:
            async def ainvoke(self, msgs, config=None):
                return type("R", (), {"content": "推荐 MC 整合包终焉之诗"})()
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot: _FakeModel())
        monkeypatch.setattr(junjun_llm, "get_callbacks", lambda: [])

        import asyncio
        dc.prewarm_video("qq:999:group", "https://v.douyin.com/FHalxoVyefQ/")
        await asyncio.gather(*list(dc._PENDING.values()), return_exceptions=True)
        await asyncio.sleep(0.05)
        block = dc.render_recent_block("qq:999:group")
        assert "最近分享的抖音" in block and "终焉之诗" in block

    def test_empty_when_nothing(self):
        assert dc.render_recent_block("qq:999:group") == ""


class TestInterceptor:
    def _ctx(self, text, url):
        return SimpleNamespace(
            args=url, meta=SimpleNamespace(text=text),
            session=SimpleNamespace(chat_id="qq:999:group"),
            reply=lambda *a, **k: None)

    @pytest.mark.asyncio
    async def test_question_hint_passes_through(self, monkeypatch):
        """「这视频讲了啥 <链接>」不消费，预热后交给 LLM。"""
        warmed = []
        monkeypatch.setattr(dc, "prewarm_video",
                            lambda cid, url: warmed.append((cid, url)))
        ctx = self._ctx("这视频讲了啥 https://v.douyin.com/abc/",
                        "https://v.douyin.com/abc/")
        consumed = await dt.douyin_hit(ctx)
        assert consumed is False
        assert warmed and warmed[0][0] == "qq:999:group"

    @pytest.mark.asyncio
    async def test_pure_share_consumed_and_prewarmed(self, monkeypatch):
        warmed = []
        monkeypatch.setattr(dc, "prewarm_video",
                            lambda cid, url: warmed.append((cid, url)))
        sent = []

        async def _submit(chat_id, url):
            return "ack"
        monkeypatch.setattr(dt, "_submit_parse", _submit)
        dt._last_use.clear()

        async def _reply(text):
            sent.append(text)
        ctx = self._ctx("快看这个 https://v.douyin.com/abc/",
                        "https://v.douyin.com/abc/")
        ctx.reply = _reply
        consumed = await dt.douyin_hit(ctx)
        assert consumed is True and warmed


class TestSummaryTool:
    @pytest.mark.asyncio
    async def test_live_message(self, monkeypatch):
        monkeypatch.setattr(dc, "get_material",
                            lambda url: _async_return({"type": "live"}))
        out = await dt.douyin_summary.ainvoke({"url": "https://v.douyin.com/xyz/"})
        assert "直播" in out

    @pytest.mark.asyncio
    async def test_fail_message(self, monkeypatch):
        monkeypatch.setattr(dc, "get_material", lambda url: _async_return(None))
        out = await dt.douyin_summary.ainvoke({"url": "https://v.douyin.com/xyz/"})
        assert "没拿到" in out


async def _async_return(v):
    return v
