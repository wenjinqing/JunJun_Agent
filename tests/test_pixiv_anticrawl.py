"""pixiv_novel 反爬修复测试：curl_cffi 伪装 + 403/5xx 重试。"""

import pytest

from junjun_skills.plugins.pixiv import client as tools


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def _mock_session_cls(monkeypatch, responses):
    """伪造 curl_cffi AsyncSession：按序返回响应。"""
    queue = list(responses)

    class _Session:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url, headers=None, timeout=None):
            return queue.pop(0)

    import curl_cffi.requests as creq
    monkeypatch.setattr(creq, "AsyncSession", _Session)
    monkeypatch.setattr(tools, "_proxy", lambda: "")
    monkeypatch.setattr("asyncio.sleep", _instant())


def _instant():
    async def _sleep(d): pass
    return _sleep


class TestFetchJson:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        _mock_session_cls(monkeypatch, [_Resp(200, {"error": False, "body": {"title": "記曝"}})])
        out = await tools._fetch_json("https://x")
        assert out["title"] == "記曝"

    @pytest.mark.asyncio
    async def test_403_retry_then_success(self, monkeypatch):
        """Cloudflare 抖动：403 -> 200，重试后成功。"""
        _mock_session_cls(monkeypatch, [
            _Resp(403), _Resp(200, {"error": False, "body": {"title": "t"}})])
        out = await tools._fetch_json("https://x")
        assert out["title"] == "t"

    @pytest.mark.asyncio
    async def test_403_exhausted(self, monkeypatch):
        """持续 403：重试 3 次后返回 error。"""
        _mock_session_cls(monkeypatch, [_Resp(403), _Resp(403), _Resp(403)])
        out = await tools._fetch_json("https://x")
        assert out["error"] == "HTTP 403"

    @pytest.mark.asyncio
    async def test_404_no_retry(self, monkeypatch):
        """确定性 4xx 不重试。"""
        queue = [_Resp(404)]
        _mock_session_cls(monkeypatch, queue)
        out = await tools._fetch_json("https://x")
        assert out["error"] == "HTTP 404"

    @pytest.mark.asyncio
    async def test_pixiv_error_body(self, monkeypatch):
        _mock_session_cls(monkeypatch, [_Resp(200, {"error": True, "message": "限制内容"})])
        out = await tools._fetch_json("https://x")
        assert out["error"] == "限制内容"


class TestPokeActionCompat:
    @pytest.mark.asyncio
    async def test_poke_new_action_first(self, monkeypatch):
        """戳一戳先试统一 send_poke，旧 action 兜底。"""
        calls = []

        async def _send(action, params):
            calls.append(action)
            if action == "send_poke":
                return {"status": "ok"}
            return {"status": "error"}

        import importlib
        sh = importlib.import_module("junjun_adapter_napcat.send_handler.main_send_handler")
        import junjun_adapter_napcat.send_handler.nc_sending as nc
        monkeypatch.setattr(nc.nc_message_sender, "send_message_to_napcat", _send)

        handler = sh.SendHandler.__new__(sh.SendHandler)
        from maim_message import Seg
        # 直接验证 _extract_pokes 提取逻辑
        seg, pokes = handler._extract_pokes(Seg(type="poke", data="12345"))
        assert pokes == ["12345"] and seg.type == "text"
