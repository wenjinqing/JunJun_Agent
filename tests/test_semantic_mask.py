"""语义工具掩码的缓存路径（2026-08-03 bug 修复）。

原实现：同步掩码路径里 `loop.run_until_complete(embed_one(...))`——生产上跑在
async 上下文，run_until_complete 在 await 之前就炸 RuntimeError，留下从未
await 的协程（RuntimeWarning 刷屏），语义掩码静默失效、永远走关键词降级。

修复：同步路径只读缓存（client.cached），缓存由 warm_tool_embeddings 在
async 上下文预热。本文件验证：缓存命中时按余弦排序补满；未命中降级关键词；
预热真的填缓存且只补未缓存的。
"""

from types import SimpleNamespace

import pytest

import junjun_memory.embedding as emb_mod
from junjun_skills import registry


def _tool(name, desc):
    return SimpleNamespace(name=name, description=desc, metadata={})


def _session(text="随便聊聊"):
    return SimpleNamespace(memory=SimpleNamespace(
        entries=[SimpleNamespace(text=text)]))


class _FakeClient:
    """预设 text->vec 的假 embedding 客户端。"""

    available = True

    def __init__(self, vecs=None):
        self._vecs = vecs or {}
        self.calls = []

    def cached(self, text):
        return self._vecs.get(text)

    async def embed_one(self, text):
        self.calls.append(text)
        self._vecs[text] = [1.0, 0.0]
        return [1.0, 0.0]


class TestCachedMaskPath:
    def test_cosine_fill_when_cache_hit(self, monkeypatch):
        """缓存命中：补满按余弦相似度排序（不靠关键词）。"""
        query = "随便聊聊"
        client = _FakeClient({
            query: [1.0, 0.0],
            "画一张图": [0.9, 0.1],      # 最相似
            "播放音乐": [0.0, 1.0],
            "设置提醒": [-1.0, 0.0],     # 最不相似
        })
        monkeypatch.setattr(emb_mod, "get_embedding_client", lambda: client)
        tools = [_tool("ai_draw", "画一张图"), _tool("play_music", "播放音乐"),
                 _tool("set_reminder", "设置提醒")]
        kept = registry._mask_by_relevance(tools, _session(query))
        assert [t.name for t in kept] == ["ai_draw", "play_music", "set_reminder"]

    def test_keyword_fallback_when_cache_miss(self, monkeypatch):
        """缓存未命中：降级关键词（同步路径绝不创建协程）。"""
        client = _FakeClient()  # 空缓存
        monkeypatch.setattr(emb_mod, "get_embedding_client", lambda: client)
        tools = [_tool("ai_draw", "画一张图"), _tool("play_music", "播放音乐")]
        kept = registry._mask_by_relevance(tools, _session("今天天气不错"))
        assert len(kept) == 2  # 不炸、不丢工具就算对

    def test_no_coroutine_created(self, monkeypatch):
        """回归：同步路径不产生未 await 协程（无 RuntimeWarning）。"""
        import warnings

        class _ExplodingClient(_FakeClient):
            async def embed_one(self, text):  # pragma: no cover
                raise AssertionError("同步路径不许调 embed_one")

        monkeypatch.setattr(emb_mod, "get_embedding_client",
                            lambda: _ExplodingClient())
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            kept = registry._mask_by_relevance(
                [_tool("ai_draw", "画一张图")], _session())
        assert len(kept) == 1


class TestWarmToolEmbeddings:
    @pytest.mark.asyncio
    async def test_warm_fills_cache(self, monkeypatch):
        client = _FakeClient()
        monkeypatch.setattr(emb_mod, "get_embedding_client", lambda: client)
        monkeypatch.setattr(registry, "_registry", {
            "ai_draw": _tool("ai_draw", "画一张图"),
            "play_music": _tool("play_music", "播放音乐"),
        })
        await registry.warm_tool_embeddings(_session("想听歌"))
        assert "画一张图" in client.calls
        assert "播放音乐" in client.calls
        assert "想听歌" in client.calls
        assert client.cached("画一张图") is not None

    @pytest.mark.asyncio
    async def test_warm_skips_cached(self, monkeypatch):
        client = _FakeClient({"画一张图": [1.0, 0.0]})
        monkeypatch.setattr(emb_mod, "get_embedding_client", lambda: client)
        monkeypatch.setattr(registry, "_registry", {
            "ai_draw": _tool("ai_draw", "画一张图"),
        })
        await registry.warm_tool_embeddings(None)
        assert client.calls == [], "已缓存的不重复打远端"

    @pytest.mark.asyncio
    async def test_warm_silent_when_unavailable(self, monkeypatch):
        client = _FakeClient()
        client.available = False
        monkeypatch.setattr(emb_mod, "get_embedding_client", lambda: client)
        await registry.warm_tool_embeddings(_session())  # 不炸即过
