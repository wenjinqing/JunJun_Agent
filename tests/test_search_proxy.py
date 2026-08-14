"""google_search 插件代理接线测试（2026-08-13 疯狂搜索事故环境层修复）。

纯离线：只测代理串解析与 config 注入，不碰网络。
"""

import pytest

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


class TestDuckDuckGoProxy:
    """2026-08-14 双实锤：①ddgs 构造器静默吞掉 config 注入的 proxy 裸连被墙
    （google/brave/wikipedia 连环超时）；②代理本身间歇性挂（7890 拒连）时
    走代理 = 全引擎硬失败，比裸连还差——失败降级直连 + 只留国内可达引擎组。"""

    def _engine(self, monkeypatch, recorded, effects, config=None):
        from junjun_skills.plugins.google_search.engines import duckduckgo as ddg

        def _fake(query, params):
            recorded.append(dict(params))
            eff = effects[min(len(recorded), len(effects)) - 1]
            if isinstance(eff, Exception):
                raise eff
            return eff

        monkeypatch.setattr(ddg, "sync_ddgs_search", _fake)
        cfg = config if config is not None else {"proxy": "http://127.0.0.1:7890"}
        return ddg.DuckDuckGoEngine(cfg)

    @pytest.mark.asyncio
    async def test_proxy_passed_to_ddgs(self, monkeypatch):
        recorded = []
        eng = self._engine(monkeypatch, recorded,
                           [[{"title": "t", "href": "u", "body": "b"}]])
        res = await eng.search("鼠标推荐", 5)
        assert len(res) == 1 and res[0].url == "u"
        assert recorded[0]["proxy"] == "http://127.0.0.1:7890"

    @pytest.mark.asyncio
    async def test_proxy_down_falls_back_direct(self, monkeypatch):
        recorded = []
        eng = self._engine(monkeypatch, recorded, [
            RuntimeError("tunnel error ... (os error 10061)"),
            [{"title": "t", "href": "u", "body": "b"}],
        ])
        res = await eng.search("鼠标推荐", 5)
        assert len(res) == 1
        assert len(recorded) == 2
        assert "proxy" not in recorded[1]          # 降级轮不带代理
        assert recorded[1]["backend"] == "duckduckgo,bing,yandex"  # 国内可达组

    @pytest.mark.asyncio
    async def test_no_results_not_retried(self, monkeypatch):
        """「没搜到」是正常空结果不是故障——不触发降级重试（误判方向守卫）。"""
        from ddgs.exceptions import DDGSException
        recorded = []
        eng = self._engine(monkeypatch, recorded,
                           [DDGSException("No results found.")])
        assert await eng.search("冷到不存在的东西", 5) == []
        assert len(recorded) == 1

    @pytest.mark.asyncio
    async def test_no_proxy_no_retry(self, monkeypatch):
        """没配代理时不存在降级空间——直连失败就失败，只试一轮。"""
        recorded = []
        eng = self._engine(monkeypatch, recorded, [RuntimeError("x")],
                           config={})
        assert await eng.search("q", 5) == []
        assert len(recorded) == 1
        assert "proxy" not in recorded[0]
