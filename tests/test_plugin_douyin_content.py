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


# ---------------------------------------------------------------- 深看（video_watch 抖音分支）

import junjun_skills.plugins.bilibili.tools as bili_tools  # noqa: E402
from junjun_skills.plugins.bilibili import watch  # noqa: E402

_DY_MATERIAL = {
    "info": {"aweme_id": "7669014018669006089", "desc": "猫猫视频", "author": "铲屎官",
             "music": "原声", "digg": 100, "comments": 20, "shares": 5,
             "duration_s": 60, "play_url": "https://aweme.snssdk.com/playwm/?video_id=abc",
             "page_url": "https://www.douyin.com/video/7669014018669006089"},
    "material": "文案：猫猫视频\n作者：铲屎官\n时长：60秒",
    "source": "文案数据",
    "page_url": "https://www.douyin.com/video/7669014018669006089",
}


class _Synth:
    """捕获综述 prompt 的假模型。"""

    def __init__(self, text="抖音观后报告"):
        self._text = text
        self.captured = ""

    async def ainvoke(self, messages, config=None):
        self.captured = str(messages[-1].content)
        return type("R", (), {"content": self._text})()


def _job():
    return type("J", (), {"job_id": "w1", "chat_id": "qq:999:group",
                          "title": "看视频", "kind": "video_watch"})()


class TestWatchDouyin:
    @pytest.fixture
    def watch_env(self, monkeypatch, tmp_path):
        """深看打桩：ffmpeg 可用、tmp 隔离、默认不抽帧。"""
        monkeypatch.setattr(bili_tools, "TMP_DIR", tmp_path)
        monkeypatch.setattr(bili_tools, "_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
        monkeypatch.setattr(watch, "_vlm_available", lambda: False)
        return monkeypatch

    @pytest.mark.asyncio
    async def test_dispatch_routes_douyin(self, watch_env, monkeypatch):
        """video_watch_handler 收到抖音链接时走抖音分支。"""
        called = {}

        async def _fake(url, **kw):
            called["url"] = url
            return "报告"

        monkeypatch.setattr(watch, "_watch_douyin", _fake)
        out = await watch.video_watch_handler(
            _job(), {"url": "https://v.douyin.com/abc/"}, synth_model=_Synth())
        assert out == "报告" and "douyin.com" in called["url"]

    @pytest.mark.asyncio
    async def test_live_rejected(self, watch_env, monkeypatch):
        monkeypatch.setattr(dc, "get_material",
                            lambda url: _async_return({"type": "live"}))
        with pytest.raises(RuntimeError, match="直播"):
            await watch.video_watch_handler(
                _job(), {"url": "https://v.douyin.com/xyz/"}, synth_model=_Synth())

    @pytest.mark.asyncio
    async def test_material_missing(self, watch_env, monkeypatch):
        monkeypatch.setattr(dc, "get_material", lambda url: _async_return(None))
        with pytest.raises(RuntimeError, match="拿不到"):
            await watch.video_watch_handler(
                _job(), {"url": "https://v.douyin.com/xyz/"}, synth_model=_Synth())

    @pytest.mark.asyncio
    async def test_too_long_rejected(self, watch_env, monkeypatch):
        import copy
        m = copy.deepcopy(_DY_MATERIAL)
        m["info"]["duration_s"] = 7200
        monkeypatch.setattr(dc, "get_material", lambda url: _async_return(m))
        with pytest.raises(RuntimeError, match="太长"):
            await watch.video_watch_handler(
                _job(), {"url": "https://v.douyin.com/abc/"}, synth_model=_Synth())

    @pytest.mark.asyncio
    async def test_full_pipeline(self, watch_env, monkeypatch, tmp_path):
        """直链下载 -> 抽音频 -> ASR -> 抽帧 -> VLM -> 综述（prompt 带「抖音」）。"""
        from pathlib import Path
        monkeypatch.setattr(dc, "get_material", lambda url: _async_return(dict(_DY_MATERIAL)))

        async def _dl(url, path):
            assert "playwm" in url
            path.write_bytes(b"\x00" * 64)
            return True

        async def _ffmpeg(args):
            if "-vn" in args:
                Path(args[-1]).write_bytes(b"audio")
            elif "-vf" in args:
                out_dir = Path(args[-1]).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "f_001.jpg").write_bytes(b"jpg")
            return True

        async def _fake_asr(path):
            return "喵喵喵转写全文"

        async def _fake_describe(data, *, model, prompt):
            return "一只猫在跳舞"

        import junjun_memory.vision as vision_mod
        monkeypatch.setattr(watch, "_download_douyin", _dl)
        monkeypatch.setattr(bili_tools, "_run_ffmpeg", _ffmpeg)
        monkeypatch.setattr(vision_mod, "_describe", _fake_describe)

        synth = _Synth()
        out = await watch.video_watch_handler(
            _job(), {"url": "https://v.douyin.com/abc/"},
            synth_model=synth, asr=_fake_asr, vlm=object())
        assert out == "抖音观后报告"
        assert "你在认真看一个抖音视频" in synth.captured
        assert "喵喵喵转写全文" in synth.captured
        assert "一只猫在跳舞" in synth.captured
        assert "猫猫视频" in synth.captured
        assert not list(tmp_path.glob("watch_dy_*"))  # 临时目录已清理

    @pytest.mark.asyncio
    async def test_no_play_url_fallback(self, watch_env, monkeypatch):
        """无直链：降级文案综述，不下载不算失败。"""
        import copy
        m = copy.deepcopy(_DY_MATERIAL)
        m["info"]["play_url"] = ""
        monkeypatch.setattr(dc, "get_material", lambda url: _async_return(m))

        async def _boom(*a, **kw):
            raise AssertionError("不该走到下载")
        monkeypatch.setattr(watch, "_download_douyin", _boom)

        out = await watch.video_watch_handler(
            _job(), {"url": "https://v.douyin.com/abc/"}, synth_model=_Synth())
        assert out == "抖音观后报告"

    @pytest.mark.asyncio
    async def test_download_failure_degrades(self, watch_env, monkeypatch):
        """下载失败：用文案材料综述兜底，job 不失败。"""
        monkeypatch.setattr(dc, "get_material", lambda url: _async_return(dict(_DY_MATERIAL)))
        monkeypatch.setattr(watch, "_download_douyin",
                            lambda url, path: _async_return(False))
        synth = _Synth()
        out = await watch.video_watch_handler(
            _job(), {"url": "https://v.douyin.com/abc/"}, synth_model=synth)
        assert out == "抖音观后报告" and "猫猫视频" in synth.captured
