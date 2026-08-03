"""语义召回「你忽然想起」（P6-1）：拟人化注入 + 每会话每小时限流。

faiss 语义检索本身早已上线（top-3/0.3 阈值/1.5s 超时），本特性是增量：
- 注入话术从干巴巴「相关记忆」改为第一人称回忆浮现（像真人）
- 每会话每小时上限（默认 5），防每轮注入稀释人设、省 embedding 调用
- 命中注入才占额度；检索失败/空结果不占
"""

from types import SimpleNamespace
from collections import deque

import pytest

import junjun_core.config.config as cfg_mod
import junjun_agent.processor as proc
import junjun_memory.long_term as lt


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"memory": {"recall_max_per_hour": 2}})
    proc._RECALL_LOG.clear()
    yield monkeypatch
    proc._RECALL_LOG.clear()
    cfg_mod.global_config = old


class _FakeLTM:
    def __init__(self, items=None, boom=False):
        self._items = items or []
        self._boom = boom
        self.calls = 0

    async def search(self, query, *, top_k=5, chat_id=None):
        self.calls += 1
        if self._boom:
            raise RuntimeError("embedding api down")
        return list(self._items)


def _item(text):
    return SimpleNamespace(text=text)


async def _block(env, monkeypatch, ltm, chat_id="qq:999:group"):
    monkeypatch.setattr(lt, "get_long_term_memory", lambda: ltm)
    session = SimpleNamespace(chat_id=chat_id, memory=None)
    meta = SimpleNamespace(image_urls=None, sticker_urls=None, voice_records=None,
                           video_urls=None, text="上次那家店叫什么来着",
                           user_id="1", nickname="甲")
    block, _ = await proc._build_memory_block(session, meta)
    return block


class TestRecallFraming:
    @pytest.mark.asyncio
    async def test_hit_injects_first_person_block(self, env, monkeypatch):
        """命中 -> 「你忽然想起」拟人块 + 占一个额度。"""
        ltm = _FakeLTM([_item("他说过不吃香菜"), _item("上周去了趟医院")])
        block = await _block(env, monkeypatch, ltm)
        assert "你忽然想起" in block
        assert "他说过不吃香菜" in block
        assert "别逐条转述" in block
        assert len(proc._RECALL_LOG["qq:999:group"]) == 1

    @pytest.mark.asyncio
    async def test_empty_result_no_slot(self, env, monkeypatch):
        """空结果不占额度。"""
        ltm = _FakeLTM([])
        block = await _block(env, monkeypatch, ltm)
        assert "你忽然想起" not in block
        assert not proc._RECALL_LOG.get("qq:999:group")

    @pytest.mark.asyncio
    async def test_failure_no_slot_and_degrades(self, env, monkeypatch):
        """检索炸 -> 降级无注入、不占额度、不阻塞。"""
        ltm = _FakeLTM(boom=True)
        block = await _block(env, monkeypatch, ltm)
        assert "你忽然想起" not in block
        assert not proc._RECALL_LOG.get("qq:999:group")


class TestRecallRateLimit:
    @pytest.mark.asyncio
    async def test_hourly_cap_skips_search(self, env, monkeypatch):
        """额度用完（本例 2/h）-> 整段跳过，连检索都不发。"""
        ltm = _FakeLTM([_item("x")])
        proc._RECALL_LOG["qq:999:group"] = deque([proc.time.time()] * 2)
        block = await _block(env, monkeypatch, ltm)
        assert "你忽然想起" not in block
        assert ltm.calls == 0

    @pytest.mark.asyncio
    async def test_cap_is_per_chat(self, env, monkeypatch):
        """限流按会话隔离。"""
        ltm = _FakeLTM([_item("x")])
        proc._RECALL_LOG["qq:999:group"] = deque([proc.time.time()] * 2)
        block = await _block(env, monkeypatch, ltm, chat_id="qq:1:private")
        assert "你忽然想起" in block

    @pytest.mark.asyncio
    async def test_zero_means_unlimited(self, env, monkeypatch):
        """recall_max_per_hour=0 -> 不限流。"""
        cfg_mod.global_config.raw["memory"]["recall_max_per_hour"] = 0
        ltm = _FakeLTM([_item("x")])
        proc._RECALL_LOG["qq:999:group"] = deque([proc.time.time()] * 99)
        block = await _block(env, monkeypatch, ltm)
        assert "你忽然想起" in block

    def test_sliding_window_expires(self, env, monkeypatch):
        """1 小时前的注入记录滑出窗口后额度恢复。"""
        now = proc.time.time()
        proc._RECALL_LOG["c"] = deque([now - 7200, now - 3700])  # 都过期
        assert proc._recall_capped("c", 2) is False
        proc._RECALL_LOG["c"].extend([now - 100, now - 50])      # 两条新鲜
        assert proc._recall_capped("c", 2) is True
