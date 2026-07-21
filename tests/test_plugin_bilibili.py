"""bilibili 插件测试：链接正则 / 命令 / 拦截器 / 信息卡降级 / 成功路径 / 压缩决策 / 限流。"""

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
    import junjun_skills.plugins.bilibili.tools as bili
    bili = importlib.reload(bili)
    bili._last_use.clear()
    return bili


def _sync_bg(monkeypatch, bili):
    """把 _spawn_bg 替换为收集协程：测试里手动 await，避免后台任务不确定时序。"""
    pending = []
    monkeypatch.setattr(bili, "_spawn_bg", lambda coro: pending.append(coro))
    return pending


async def _drain(pending):
    while pending:
        await pending.pop(0)


_VIEW = {
    "bvid": "BV1xx411c7mD", "aid": 17001, "cid": 280001,
    "title": "猫猫弹琴", "desc": "一只会弹琴的猫",
    "duration": 100, "pic": "http://i0.hdslb.com/cover.jpg", "owner": "UP主甲",
}


_UNSET = object()


def _patch_common(monkeypatch, bili, *, view=_UNSET, playurl=_UNSET,
                  ffmpeg="/usr/bin/ffmpeg", extract=True):
    """打公共桩：BV 提取 / view / playurl API 与 ffmpeg 探测。"""
    if extract:
        async def _extract(url):
            return "BV1xx411c7mD"

        monkeypatch.setattr(bili, "extract_bvid", _extract)

    async def _view(bvid):
        v = _VIEW if view is _UNSET else view
        return dict(v) if isinstance(v, dict) else v

    async def _play(aid, cid):
        return {"type": "durl", "url": "https://cdn.example/v.mp4"} if playurl is _UNSET else playurl

    monkeypatch.setattr(bili, "_fetch_view", _view)
    monkeypatch.setattr(bili, "_fetch_playurl", _play)
    monkeypatch.setattr(bili, "_ffmpeg_path", lambda: ffmpeg)


def _fake_download(monkeypatch, bili, size=1024):
    """假下载：直接写本地文件，返回成功。"""
    async def _dl(url, path):
        path.write_bytes(b"\x00" * size)
        return True

    monkeypatch.setattr(bili, "_download", _dl)


class TestUrlRegex:
    def test_hit(self):
        bili = _load()
        assert bili._first_bili_url("快看 https://www.bilibili.com/video/BV1xx411c7mD 好逗")
        assert bili._first_bili_url("https://b23.tv/abc123")
        assert bili._first_bili_url("https://m.bilibili.com/video/BV1xx411c7mD?p=2")
        # 尾随标点应被剥掉
        assert bili._first_bili_url("https://b23.tv/abc123。") == "https://b23.tv/abc123"

    def test_miss(self):
        bili = _load()
        assert bili._first_bili_url("没有链接哦") is None
        assert bili._first_bili_url("https://v.douyin.com/abc/") is None
        assert bili._first_bili_url("") is None

    @pytest.mark.asyncio
    async def test_extract_bvid_short_link(self, monkeypatch):
        """b23 短链先跟随重定向再提取 BV 号。"""
        bili = _load()

        async def _redirect(url):
            return "https://www.bilibili.com/video/BV1xx411c7mD?spm_id=xx"

        monkeypatch.setattr(bili, "_follow_redirect", _redirect)
        assert await bili.extract_bvid("https://b23.tv/abc123") == "BV1xx411c7mD"
        # 标准链接不走重定向
        assert await bili.extract_bvid("https://www.bilibili.com/video/BV1xx411c7mD") == "BV1xx411c7mD"
        assert await bili.extract_bvid("https://example.com/x") is None


class TestCommand:
    @pytest.mark.asyncio
    async def test_usage_and_bad_link(self):
        bili = _load()
        assert "用法" in await bili.bilibili_cmd(_ctx("/bilibili"))
        assert "有效" in await bili.bilibili_cmd(_ctx("/bilibili https://example.com/x"))

    @pytest.mark.asyncio
    async def test_success_sends_video_and_cleans(self, _fake_gateway, monkeypatch, tmp_path):
        """成功路径：假视频文件 -> 发 video 段 + 标题文本，临时文件发送后删除。"""
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili)
        created = []

        async def _dl(url, path):
            path.write_bytes(b"\x00" * 1024)
            created.append(path)
            return True

        monkeypatch.setattr(bili, "_download", _dl)

        result = await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1xx411c7mD"))
        assert result is None
        assert _fake_gateway[0].segments[0].data == "开始解析 B 站视频，请稍候～"
        await _drain(pending)

        segs = _fake_gateway[1].segments
        assert segs[0].type == "text" and "猫猫弹琴" in segs[0].data
        assert segs[1].type == "video"
        assert segs[1].data.endswith(".mp4")
        # 临时文件已清理
        assert created and not created[0].exists()

    @pytest.mark.asyncio
    async def test_dash_merge_path(self, _fake_gateway, monkeypatch, tmp_path):
        """DASH：下载音视频流后调 _ffmpeg_merge，合并成功发 video 段。"""
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili, playurl={
            "type": "dash", "video": "https://cdn.example/v.m4s", "audio": "https://cdn.example/a.m4s"})
        _fake_download(monkeypatch, bili)
        merges = []

        async def _merge(v, a, out):
            merges.append((v, a, out))
            out.write_bytes(b"\x00" * 1024)
            return True

        monkeypatch.setattr(bili, "_ffmpeg_merge", _merge)
        await bili.bilibili_cmd(_ctx("/b站 https://www.bilibili.com/video/BV1xx411c7mD"))
        await _drain(pending)

        assert len(merges) == 1 and merges[0][1] is not None  # 音轨下载成功则参与合并
        segs = _fake_gateway[1].segments
        assert any(s.type == "video" for s in segs)
        assert not list(tmp_path.iterdir())  # 临时目录已清空

    @pytest.mark.asyncio
    async def test_no_ffmpeg_info_card(self, _fake_gateway, monkeypatch, tmp_path):
        """无 ffmpeg：只发信息卡（标题/UP主/简介/时长/封面 image/链接），不下载。"""
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili, ffmpeg=None)

        async def _dl(url, path):  # 不应被调用
            raise AssertionError("无 ffmpeg 不应下载")

        monkeypatch.setattr(bili, "_download", _dl)
        await bili.bilibili_cmd(_ctx("/bilibili https://b23.tv/abc123"))
        await _drain(pending)

        segs = _fake_gateway[1].segments
        assert segs[0].type == "text"
        text = segs[0].data
        assert "猫猫弹琴" in text and "UP主甲" in text and "1分40秒" in text
        assert "https://www.bilibili.com/video/BV1xx411c7mD" in text
        assert segs[1].type == "image" and segs[1].data == "http://i0.hdslb.com/cover.jpg"
        assert not any(s.type == "video" for s in segs)

    @pytest.mark.asyncio
    async def test_over_duration_compress_ok(self, _fake_gateway, monkeypatch, tmp_path):
        """超时长：触发压缩决策，压缩后大小达标则发 video 段。"""
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili, view={**_VIEW, "duration": 9999})
        _fake_download(monkeypatch, bili, size=1024)
        calls = []

        async def _compress(src, out):
            calls.append(src)
            out.write_bytes(b"\x00" * 512)
            return True

        monkeypatch.setattr(bili, "_ffmpeg_compress", _compress)
        await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1xx411c7mD"))
        await _drain(pending)

        assert len(calls) == 1  # 超时长触发了压缩
        segs = _fake_gateway[1].segments
        assert any(s.type == "video" for s in segs)
        assert not list(tmp_path.iterdir())

    @pytest.mark.asyncio
    async def test_over_duration_compress_fail_info_card(self, _fake_gateway, monkeypatch, tmp_path):
        """超时长且压缩失败：降级信息卡 + 链接，不发 video 段。"""
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili, view={**_VIEW, "duration": 9999})
        _fake_download(monkeypatch, bili)

        async def _compress(src, out):
            return False

        monkeypatch.setattr(bili, "_ffmpeg_compress", _compress)
        await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1xx411c7mD"))
        await _drain(pending)

        segs = _fake_gateway[1].segments
        assert segs[0].type == "text" and "超过限制" in segs[0].data
        assert not any(s.type == "video" for s in segs)
        assert not list(tmp_path.iterdir())

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, monkeypatch, tmp_path):
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili)
        _fake_download(monkeypatch, bili)
        calls = []

        async def _extract(url):
            calls.append(url)
            return "BV1xx411c7mD"

        monkeypatch.setattr(bili, "extract_bvid", _extract)
        await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1xx411c7mD"))
        result = await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1yy411c7mE"))
        assert "秒后再试" in result
        await _drain(pending)
        assert len(calls) == 1  # 冷却内不再解析
        # 换个会话不受限
        await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1yy411c7mE", is_group=False))
        await _drain(pending)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_view_failure_friendly_text(self, _fake_gateway, monkeypatch):
        bili = _load()
        pending = _sync_bg(monkeypatch, bili)

        async def _view(bvid):
            return None

        monkeypatch.setattr(bili, "_fetch_view", _view)
        await bili.bilibili_cmd(_ctx("/bilibili https://www.bilibili.com/video/BV1xx411c7mD"))
        await _drain(pending)
        assert "失败" in _fake_gateway[1].segments[0].data


class TestInterceptor:
    @pytest.mark.asyncio
    async def test_dispatch_hit_consumes(self, _fake_gateway, monkeypatch, tmp_path):
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili)
        _fake_download(monkeypatch, bili)

        meta = _meta("这个好看 https://www.bilibili.com/video/BV1xx411c7mD 哈哈哈")
        consumed = await interceptors.dispatch(_session(), meta)
        assert consumed is True
        await _drain(pending)
        assert any(s.type == "video" for s in _fake_gateway[1].segments)

    @pytest.mark.asyncio
    async def test_dispatch_short_link_hit(self, _fake_gateway, monkeypatch, tmp_path):
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili, extract=False)  # 保留真实 extract_bvid 以验证短链重定向
        _fake_download(monkeypatch, bili)
        seen = []

        async def _redirect(url):
            seen.append(url)
            return "https://www.bilibili.com/video/BV1xx411c7mD"

        monkeypatch.setattr(bili, "_follow_redirect", _redirect)
        consumed = await interceptors.dispatch(_session(), _meta("https://b23.tv/abc123"))
        assert consumed is True
        await _drain(pending)
        assert seen  # 短链走了重定向

    @pytest.mark.asyncio
    async def test_dispatch_miss_passes(self, _fake_gateway):
        _load()
        consumed = await interceptors.dispatch(_session(), _meta("今天天气不错"))
        assert consumed is False
        assert not _fake_gateway

    @pytest.mark.asyncio
    async def test_hit_rate_limited(self, _fake_gateway, monkeypatch, tmp_path):
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili)
        _fake_download(monkeypatch, bili)

        assert await bili.bilibili_hit(_hit_ctx("https://www.bilibili.com/video/BV1xx411c7mD")) is True
        await _drain(pending)
        _fake_gateway.clear()
        assert await bili.bilibili_hit(_hit_ctx("https://www.bilibili.com/video/BV1yy411c7mE")) is True
        assert "秒后再试" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_playurl_failure_info_card(self, _fake_gateway, monkeypatch, tmp_path):
        """playurl 失败：降级信息卡 + 链接。"""
        bili = _load()
        monkeypatch.setattr(bili, "TMP_DIR", tmp_path)
        pending = _sync_bg(monkeypatch, bili)
        _patch_common(monkeypatch, bili, playurl=None)

        consumed = await bili.bilibili_hit(_hit_ctx("https://www.bilibili.com/video/BV1xx411c7mD"))
        assert consumed is True
        await _drain(pending)
        segs = _fake_gateway[1].segments
        assert segs[0].type == "text" and "播放地址获取失败" in segs[0].data
        assert segs[1].type == "image"


class TestManifest:
    def test_manifest_and_registration(self):
        import json
        from pathlib import Path
        bili = _load()
        manifest = json.loads(
            (Path(bili.__file__).parent / "_manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == "bilibili"
        assert manifest["tools_attr"] == "TOOLS"
        assert bili.TOOLS == []
        # 命令/拦截器的 plugin 参数与 manifest name 一致
        cmd_plugins = {c["name"]: c["plugin"] for c in commands.list_commands()}
        it_plugins = {i["name"]: i["plugin"] for i in interceptors.list_interceptors()}
        assert cmd_plugins.get("bilibili") == "bilibili"
        assert it_plugins.get("bilibili_link") == "bilibili"
