"""netdisk 插件测试：链接正则 / 解析成功 / 待补密码重试 / 状态过期 / 失败降级 / 限流。"""

import time
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
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:12345:private")


def _meta(text, user_id="12345"):
    return SimpleNamespace(text=text, user_id=user_id, nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text):
    return commands.CommandContext(session=_session(), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


def _ok(link="http://dl.example.com/f.zip"):
    return {"ok": True, "link": link, "err": "", "server_error": False}


def _fail(err, server_error=False):
    return {"ok": False, "link": "", "err": err, "server_error": server_error}


class TestLinkRegex:
    def test_supported_links(self):
        import junjun_skills.plugins.netdisk.tools as nd
        for url in ("https://pan.baidu.com/s/1abc", "https://www.123pan.com/s/xyz",
                    "https://wwp.lanzoup.com/iAbc123", "https://cowtransfer.com/s/abc",
                    "https://115.com/s/xxx", "https://www.feijipan.com/s/yyy"):
            assert nd.NETDISK_URL_RE.search(url), url
        assert not nd.NETDISK_URL_RE.search("https://www.example.com/x")

    def test_first_url_strips_punct_and_extract_pwd(self):
        import junjun_skills.plugins.netdisk.tools as nd
        text = "看这个 https://pan.baidu.com/s/1abc。提取码: ab12"
        url = nd._first_netdisk_url(text)
        assert url == "https://pan.baidu.com/s/1abc"
        assert nd._extract_pwd(text, url) == "ab12"
        assert nd._extract_pwd("快存 https://pan.baidu.com/s/1abc", url) == ""


class TestCommand:
    @pytest.mark.asyncio
    async def test_usage_and_invalid_link(self):
        import junjun_skills.plugins.netdisk.tools as nd
        assert "用法" in await nd.netdisk_cmd(_ctx("/netdisk"))
        assert "有效" in await nd.netdisk_cmd(_ctx("/netdisk https://www.example.com/a"))

    @pytest.mark.asyncio
    async def test_success_sends_direct_link(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        calls = []

        async def _fetch(url, pwd=""):
            calls.append((url, pwd))
            return _ok()

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        result = await nd.netdisk_cmd(_ctx("/netdisk https://www.123pan.com/s/abc ab12"))
        assert result is None  # 已自行 reply
        assert calls == [("https://www.123pan.com/s/abc", "ab12")]
        text = _fake_gateway[0].segments[0].data
        assert "直链" in text and "http://dl.example.com/f.zip" in text


class TestAutoParse:
    @pytest.mark.asyncio
    async def test_group_without_at_auto_parses(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._last_parse.clear()

        async def _fetch(url, pwd=""):
            return _ok()

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        consumed = await nd.netdisk_link_hit(_ctx("看这个 https://cowtransfer.com/s/abc 好东西"))
        assert consumed is True
        assert "http://dl.example.com/f.zip" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_no_link_passes(self):
        import junjun_skills.plugins.netdisk.tools as nd
        assert await nd.netdisk_link_hit(_ctx("今天天气真好")) is False

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._last_parse.clear()
        calls = []

        async def _fetch(url, pwd=""):
            calls.append(url)
            return _ok()

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        assert await nd.netdisk_link_hit(_ctx("https://www.123pan.com/s/a")) is True
        assert await nd.netdisk_link_hit(_ctx("https://www.123pan.com/s/b")) is True  # 限流内静默消费
        assert len(calls) == 1


class TestPendingPwd:
    @pytest.mark.asyncio
    async def test_missing_pwd_then_retry_success(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._pending_pwd.clear()
        nd._last_parse.clear()
        calls = []

        async def _fetch(url, pwd=""):
            calls.append(pwd)
            return _ok() if pwd else _fail("需要提取码")

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        # 第一条：发链接，缺密码 → 记待补状态并提示
        assert await nd.netdisk_link_hit(_ctx("https://pan.baidu.com/s/1abc")) is True
        assert "提取码" in _fake_gateway[0].segments[0].data
        key = ("qq:999:group", "12345")
        assert key in nd._pending_pwd
        # 第二条：补发提取码 → 自动重试成功并清状态
        assert await nd.netdisk_pwd_hit(_ctx("ab12")) is True
        assert calls == ["", "ab12"]
        assert key not in nd._pending_pwd
        assert "http://dl.example.com/f.zip" in _fake_gateway[1].segments[0].data

    @pytest.mark.asyncio
    async def test_server_error_also_asks_pwd(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._pending_pwd.clear()
        nd._last_parse.clear()

        async def _fetch(url, pwd=""):
            return _fail("解析服务异常（HTTP 500）", server_error=True)

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        assert await nd.netdisk_link_hit(_ctx("https://pan.baidu.com/s/1abc")) is True
        assert "提取码" in _fake_gateway[0].segments[0].data
        assert ("qq:999:group", "12345") in nd._pending_pwd

    @pytest.mark.asyncio
    async def test_pwd_without_pending_passes(self):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._pending_pwd.clear()
        assert await nd.netdisk_pwd_hit(_ctx("ab12")) is False  # 无待补状态，放行

    @pytest.mark.asyncio
    async def test_pending_expired(self, _fake_gateway):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._pending_pwd.clear()
        key = ("qq:999:group", "12345")
        nd._pending_pwd[key] = {"url": "https://pan.baidu.com/s/1abc",
                                "ts": time.time() - nd._PENDING_TTL - 10}
        assert await nd.netdisk_pwd_hit(_ctx("ab12")) is False  # 过期放行
        assert key not in nd._pending_pwd


class TestFailureDegrade:
    @pytest.mark.asyncio
    async def test_plain_failure_friendly_text(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._pending_pwd.clear()
        nd._last_parse.clear()

        async def _fetch(url, pwd=""):
            return _fail("链接已失效或不存在")

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        assert await nd.netdisk_link_hit(_ctx("https://www.123pan.com/s/dead")) is True
        text = _fake_gateway[0].segments[0].data
        assert "没扒动" in text and "链接已失效" in text
        assert nd._pending_pwd == {}  # 非密码问题不记待补状态

    @pytest.mark.asyncio
    async def test_service_unreachable_no_pending(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.netdisk.tools as nd
        nd._pending_pwd.clear()
        nd._last_parse.clear()

        async def _fetch(url, pwd=""):
            return _fail("无法连接解析服务")

        monkeypatch.setattr(nd, "fetch_direct_link", _fetch)
        assert await nd.netdisk_link_hit(_ctx("https://www.123pan.com/s/x")) is True
        assert "没扒动" in _fake_gateway[0].segments[0].data
        assert nd._pending_pwd == {}


class TestRegistration:
    def test_command_and_interceptors_registered(self):
        import importlib
        import junjun_skills.plugins.netdisk.tools as nd
        importlib.reload(nd)  # 总线被 fixture 清空，重载模块触发装饰器重新注册
        cmds = {c["name"]: c for c in commands.list_commands()}
        assert cmds["netdisk"]["plugin"] == "netdisk"
        its = {i["name"]: i for i in interceptors.list_interceptors()}
        assert its["netdisk_link"]["plugin"] == "netdisk"
        assert its["netdisk_pwd"]["plugin"] == "netdisk"

    def test_manifest_matches(self):
        import json
        from pathlib import Path
        import junjun_skills.plugins.netdisk.tools as nd
        manifest = json.loads(
            (Path(nd.__file__).parent / "_manifest.json").read_text(encoding="utf-8"))
        assert manifest["name"] == "netdisk"
        assert manifest["module"] == "junjun_skills.plugins.netdisk.tools"
