"""google_search 插件代理接线测试（2026-08-13 疯狂搜索事故环境层修复）。

纯离线：只测代理串解析与 config 注入，不碰网络。
"""

from junjun_skills.plugins.google_search import tools as gst


class TestNormalizeProxy:
    def test_plain_host_port(self):
        assert gst._normalize_proxy("127.0.0.1:7890") == "http://127.0.0.1:7890"

    def test_per_protocol_string_prefers_https(self):
        s = "http=127.0.0.1:7891;https=127.0.0.1:7892"
        assert gst._normalize_proxy(s) == "http://127.0.0.1:7892"

    def test_already_a_url(self):
        assert gst._normalize_proxy("http://127.0.0.1:7890") == "http://127.0.0.1:7890"
        assert gst._normalize_proxy("socks5://127.0.0.1:1080") == "socks5://127.0.0.1:1080"

    def test_empty_and_garbage(self):
        assert gst._normalize_proxy("") == ""
        assert gst._normalize_proxy("   ") == ""
        assert gst._normalize_proxy("=") == ""


class TestEngineProxyWiring:
    def test_build_engine_injects_proxy(self, monkeypatch):
        monkeypatch.setattr(gst, "SEARCH_PROXY", "http://127.0.0.1:7890")
        captured = {}

        class _FakeEngine:
            def __init__(self, config=None):
                captured.update(config or {})

        monkeypatch.setitem(gst.ENGINE_MAP, "bing", _FakeEngine)
        gst._build_engine("bing")
        assert captured.get("proxy") == "http://127.0.0.1:7890"

    def test_no_proxy_no_key(self, monkeypatch):
        """未配代理时 config 不带 proxy 键（引擎回退自身逻辑，不塞空串）。"""
        monkeypatch.setattr(gst, "SEARCH_PROXY", "")
        captured = {}

        class _FakeEngine:
            def __init__(self, config=None):
                captured.update(config or {})

        monkeypatch.setitem(gst.ENGINE_MAP, "sogou", _FakeEngine)
        gst._build_engine("sogou")
        assert "proxy" not in captured
