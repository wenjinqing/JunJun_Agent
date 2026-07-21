"""douyin 插件测试：链接正则 / 命令 / 拦截器 / 解析降级 / 限流。"""

import importlib
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


def _hit_ctx(url, is_group=True):
    """拦截器上下文：args 为正则命中的链接串。"""
    return commands.CommandContext(session=_session(is_group), meta=_meta(url), args=url)


def _load():
    """导入（并重载）插件模块：清总线后重新触发装饰器注册，并清空限流 dict。"""
    import junjun_skills.plugins.douyin.tools as dy
    dy = importlib.reload(dy)
    dy._last_use.clear()
    return dy


_VIDEO_PAYLOAD = {
    "code": 200,
    "data": {
        "item": {"title": "猫猫视频", "author": "铲屎官", "url": "http://cdn.example/v.mp4"},
        "stat": {"like": 100, "comment": 20, "collect": 5, "share": 3},
    },
}

_GALLERY_PAYLOAD = {
    "code": 200,
    "data": {
        "item": {"title": "美景图集", "author": "摄影师",
                 "images": [f"http://cdn.example/{i}.jpg" for i in range(12)]},
        "stat": {"like": 9, "comment": 8, "collect": 7, "share": 6},
    },
}


class TestUrlRegex:
    def test_hit(self):
        dy = _load()
        assert dy._first_douyin_url("快看 https://v.douyin.com/UW8-u_REUP8/ 好逗")
        assert dy._first_douyin_url("https://www.douyin.com/video/7123456789012345678")
        assert dy._first_douyin_url("https://www.douyin.com/note/abc-def_123")
        # 尾随标点应被剥掉
        assert dy._first_douyin_url("https://v.douyin.com/abc/。") == "https://v.douyin.com/abc/"

    def test_miss(self):
        dy = _load()
        assert dy._first_douyin_url("没有链接哦") is None
        assert dy._first_douyin_url("https://www.bilibili.com/video/BV1xx") is None
        assert dy._first_douyin_url("") is None


class TestCommand:
    @pytest.mark.asyncio
    async def test_usage_and_bad_link(self):
        dy = _load()
        assert "用法" in await dy.douyin_cmd(_ctx("/douyin"))
        assert "有效" in await dy.douyin_cmd(_ctx("/douyin https://example.com/x"))

    @pytest.mark.asyncio
    async def test_video_success_summary(self, _fake_gateway, monkeypatch):
        dy = _load()

        async def _fetch(url):
            return _VIDEO_PAYLOAD

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        result = await dy.douyin_cmd(_ctx("/douyin https://v.douyin.com/abc/"))
        assert result is None
        segs = _fake_gateway[0].segments
        assert segs[0].type == "text"
        assert "猫猫视频" in segs[0].data and "铲屎官" in segs[0].data
        assert "❤️100" in segs[0].data and "💬20" in segs[0].data
        assert "📎 视频：http://cdn.example/v.mp4" in segs[0].data

    @pytest.mark.asyncio
    async def test_gallery_sends_images_capped(self, _fake_gateway, monkeypatch):
        dy = _load()

        async def _fetch(url):
            return _GALLERY_PAYLOAD

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        await dy.douyin_cmd(_ctx("/抖音解析 https://www.douyin.com/note/abc123"))
        segs = _fake_gateway[0].segments
        assert segs[0].type == "text" and "美景图集" in segs[0].data
        images = [s for s in segs if s.type == "image"]
        assert len(images) == 9  # 上限 9 张
        assert images[0].data == "http://cdn.example/0.jpg"

    @pytest.mark.asyncio
    async def test_jx_nested_payload(self, _fake_gateway, monkeypatch):
        """兼容 jx[0] 内嵌 item/stat 的多层结构。"""
        dy = _load()

        async def _fetch(url):
            return {"msg": "解析成功",
                    "data": {"jx": [{"item": {"title": "嵌套", "url": "http://x/v.mp4"},
                                     "stat": {"like": 1}}]}}

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        await dy.douyin_cmd(_ctx("/douyin https://v.douyin.com/abc/"))
        data = _fake_gateway[0].segments[0].data
        assert "嵌套" in data and "http://x/v.mp4" in data

    @pytest.mark.asyncio
    async def test_fetch_failure_degrades(self, monkeypatch):
        dy = _load()

        async def _none(url):
            return None

        monkeypatch.setattr(dy, "_fetch_parse", _none)
        result = await dy.douyin_cmd(_ctx("/douyin https://v.douyin.com/abc/"))
        assert result is None or "用法" not in (result or "")
        # 失败走 ctx.reply 友好文本（无 gateway 时不抛异常即降级成功）

    @pytest.mark.asyncio
    async def test_api_failure_msg(self, _fake_gateway, monkeypatch):
        dy = _load()

        async def _fetch(url):
            return {"code": 400, "msg": "解析失败：资源id获取失败"}

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        await dy.douyin_cmd(_ctx("/douyin https://v.douyin.com/abc/"))
        assert "解析失败" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, monkeypatch):
        dy = _load()
        calls = []

        async def _fetch(url):
            calls.append(url)
            return _VIDEO_PAYLOAD

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        await dy.douyin_cmd(_ctx("/douyin https://v.douyin.com/abc/"))
        result = await dy.douyin_cmd(_ctx("/douyin https://v.douyin.com/def/"))
        assert "秒后再试" in result
        assert len(calls) == 1  # 冷却内不再请求接口
        # 换个会话不受限
        ctx = _ctx("/douyin https://v.douyin.com/def/", is_group=False)
        await dy.douyin_cmd(ctx)
        assert len(calls) == 2


class TestInterceptor:
    @pytest.mark.asyncio
    async def test_dispatch_hit_consumes(self, _fake_gateway, monkeypatch):
        dy = _load()

        async def _fetch(url):
            return _VIDEO_PAYLOAD

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        meta = _meta("这个好看 https://v.douyin.com/UW8-u_REUP8/ 哈哈哈")
        consumed = await interceptors.dispatch(_session(), meta)
        assert consumed is True
        assert "猫猫视频" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_dispatch_miss_passes(self, _fake_gateway):
        _load()
        consumed = await interceptors.dispatch(_session(), _meta("今天天气不错"))
        assert consumed is False
        assert not _fake_gateway

    @pytest.mark.asyncio
    async def test_hit_failure_friendly_text(self, _fake_gateway, monkeypatch):
        dy = _load()

        async def _none(url):
            return None

        monkeypatch.setattr(dy, "_fetch_parse", _none)
        consumed = await dy.douyin_hit(_hit_ctx("https://v.douyin.com/abc/"))
        assert consumed is True
        assert "稍后再试" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_hit_rate_limited(self, _fake_gateway, monkeypatch):
        dy = _load()

        async def _fetch(url):
            return _VIDEO_PAYLOAD

        monkeypatch.setattr(dy, "_fetch_parse", _fetch)
        assert await dy.douyin_hit(_hit_ctx("https://v.douyin.com/abc/")) is True
        _fake_gateway.clear()
        assert await dy.douyin_hit(_hit_ctx("https://v.douyin.com/def/")) is True
        assert "秒后再试" in _fake_gateway[0].segments[0].data


class TestManifest:
    def test_manifest_and_registration(self):
        import json
        from pathlib import Path
        dy = _load()
        manifest = json.loads(
            (Path(dy.__file__).parent / "_manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == "douyin"
        assert manifest["tools_attr"] == "TOOLS"
        assert dy.TOOLS == []
        # 命令/拦截器的 plugin 参数与 manifest name 一致
        cmd_plugins = {c["name"]: c["plugin"] for c in commands.list_commands()}
        it_plugins = {i["name"]: i["plugin"] for i in interceptors.list_interceptors()}
        assert cmd_plugins.get("douyin") == "douyin"
        assert it_plugins.get("douyin_link") == "douyin"
