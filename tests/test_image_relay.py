"""ai_draw → send_feed 图片接力测试：「画图发空间」只画一张图。

覆盖：
- wait_recent_drawn_url：无记录 / 等进行中的画图 / 30s 内完成图复用 / 过期不复用
- _feed_image_bytes：有接力图不重复生成 / 无接力图自己生成 / 生成失败降级 None
"""

import asyncio
import time

import pytest

from junjun_skills.plugins.ai_draw import tools as ad
from junjun_skills.plugins.junzone import tools as mz

CHAT = "qq:1158561385:group"
IMG_URL = "http://img.example/abc.png"
FAKE_BYTES = b"\x89PNG" + b"x" * 2000


@pytest.fixture(autouse=True)
def _clean_relay_state():
    ad._PENDING.clear()
    ad._LAST_DRAWN.clear()
    yield
    ad._PENDING.clear()
    ad._LAST_DRAWN.clear()


def _fake_pipeline(url=IMG_URL, delay=0.0):
    async def _run(prompt):
        if delay:
            await asyncio.sleep(delay)
        return url, "final prompt"
    return _run


class TestWaitRecentDrawn:
    @pytest.mark.asyncio
    async def test_empty_chat_id(self):
        assert await ad.wait_recent_drawn_url("") is None

    @pytest.mark.asyncio
    async def test_no_record_returns_none(self):
        assert await ad.wait_recent_drawn_url(CHAT, timeout=0.1) is None

    @pytest.mark.asyncio
    async def test_waits_for_inflight_draw(self, monkeypatch):
        """画图进行中：send_feed 侧等待并拿到同一张图 URL。"""
        monkeypatch.setattr(ad, "_draw_pipeline", _fake_pipeline(delay=0.05))
        fut = ad._begin_pending_draw(CHAT)
        task = asyncio.create_task(ad._draw_work("猫娘", CHAT, fut))
        url = await ad.wait_recent_drawn_url(CHAT, timeout=2)
        await task
        assert url == IMG_URL
        # 完成后 pending 弹出、完成缓存写入
        assert CHAT not in ad._PENDING
        assert CHAT in ad._LAST_DRAWN

    @pytest.mark.asyncio
    async def test_reuse_within_grace_window(self, monkeypatch):
        """画图刚好完成（pending 已弹出）：30s 窗口内仍可复用。"""
        monkeypatch.setattr(ad, "_draw_pipeline", _fake_pipeline())
        fut = ad._begin_pending_draw(CHAT)
        await ad._draw_work("猫娘", CHAT, fut)
        assert await ad.wait_recent_drawn_url(CHAT, timeout=0.1) == IMG_URL

    @pytest.mark.asyncio
    async def test_stale_cache_not_reused(self):
        """更早（>30s）别人画的图绝不复用——防张冠李戴。"""
        ad._LAST_DRAWN[CHAT] = (time.time() - 60, IMG_URL)
        assert await ad.wait_recent_drawn_url(CHAT, timeout=0.1) is None

    @pytest.mark.asyncio
    async def test_failed_draw_returns_none(self, monkeypatch):
        """画图失败：等待方拿到 None（走自己生成的兜底），不挂住。"""
        monkeypatch.setattr(ad, "_draw_pipeline", _fake_pipeline(url=None))
        fut = ad._begin_pending_draw(CHAT)
        task = asyncio.create_task(ad._draw_work("猫娘", CHAT, fut))
        assert await ad.wait_recent_drawn_url(CHAT, timeout=2) is None
        assert await task is None
        assert CHAT not in ad._PENDING
        assert CHAT not in ad._LAST_DRAWN

    @pytest.mark.asyncio
    async def test_pipeline_exception_unblocks_waiter(self, monkeypatch):
        """画图抛异常：Future 回填 None，等待方不挂死。"""
        async def _boom(prompt):
            raise RuntimeError("api down")
        monkeypatch.setattr(ad, "_draw_pipeline", _boom)
        fut = ad._begin_pending_draw(CHAT)
        task = asyncio.create_task(ad._draw_work("猫娘", CHAT, fut))
        assert await ad.wait_recent_drawn_url(CHAT, timeout=2) is None
        with pytest.raises(RuntimeError):
            await task
        assert CHAT not in ad._PENDING


class TestFeedImageBytes:
    @pytest.mark.asyncio
    async def test_reuse_recent_draw_no_regenerate(self, monkeypatch):
        """本会话刚画了图：说说直接复用，绝不二次调用画图管线。"""
        from junjun_skills.builtin.memory_skills import current_chat_id

        async def _boom(prompt):
            raise AssertionError("不应重复生成图片")
        monkeypatch.setattr(ad, "_draw_pipeline", _boom)

        async def _fake_dl(url):
            assert url == IMG_URL
            return FAKE_BYTES
        monkeypatch.setattr(mz, "_download_image_bytes", _fake_dl)

        token = current_chat_id.set(CHAT)
        ad._LAST_DRAWN[CHAT] = (time.time(), IMG_URL)
        try:
            assert await mz._feed_image_bytes("正文") == FAKE_BYTES
        finally:
            current_chat_id.reset(token)

    @pytest.mark.asyncio
    async def test_no_relay_generates_own(self, monkeypatch):
        """无接力图（如定时自动说说）：正常走画图管线生成。"""
        calls = []

        async def _pipeline(prompt):
            calls.append(prompt)
            return IMG_URL, "p"
        monkeypatch.setattr(ad, "_draw_pipeline", _pipeline)

        async def _fake_dl(url):
            return FAKE_BYTES
        monkeypatch.setattr(mz, "_download_image_bytes", _fake_dl)

        # 不设置 current_chat_id（无会话上下文，如定时任务）
        assert await mz._feed_image_bytes("正文") == FAKE_BYTES
        assert calls == ["正文"]

    @pytest.mark.asyncio
    async def test_generate_failure_degrades_none(self, monkeypatch):
        monkeypatch.setattr(ad, "_draw_pipeline", _fake_pipeline(url=None))
        assert await mz._feed_image_bytes("正文") is None
