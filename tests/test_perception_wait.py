"""P0-14 感知就绪等待测试：describe_images/describe_stickers 有界等待。

- 快任务：正常拿到描述
- 慢任务：到点降级占位，但在途任务【不取消】继续跑
- wait<=0：全部等完（旧行为）
- 多张图：完成的用结果、没完成的占位（互不影响）
"""

import asyncio
import time

import pytest

from junjun_memory import vision


def _mk_task_factory(delay: float, prefix: str = "描述"):
    """describe_image_shared 替身：返回 delay 秒后才完成的任务。"""
    def _mk(url, **kw):
        async def _run():
            await asyncio.sleep(delay)
            return f"{prefix}:{url}"
        return asyncio.create_task(_run())
    return _mk


class TestBoundedWait:
    @pytest.mark.asyncio
    async def test_fast_task_returns_description(self, monkeypatch):
        monkeypatch.setattr(vision, "describe_image_shared", _mk_task_factory(0.01))
        out = await vision.describe_images(["http://x/1.png"], model=object(), wait=1.0)
        assert out == {"http://x/1.png": "描述:http://x/1.png"}

    @pytest.mark.asyncio
    async def test_slow_task_degrades_to_placeholder(self, monkeypatch):
        """VLM 慢（5s）：到 0.05s 上限降级占位，决策不被拖住。"""
        monkeypatch.setattr(vision, "describe_image_shared", _mk_task_factory(5.0))
        start = time.monotonic()
        out = await vision.describe_images(["http://x/1.png"], model=object(), wait=0.05)
        elapsed = time.monotonic() - start
        assert out == {"http://x/1.png": "[图片]"}
        assert elapsed < 1.0  # 没有等慢任务
        # 清理在途任务，防泄漏告警
        await asyncio.sleep(0)
        for t in list(getattr(vision, "_PENDING", {}).values()):
            t.cancel()

    @pytest.mark.asyncio
    async def test_slow_task_not_cancelled(self, monkeypatch):
        """降级≠放弃：在途任务继续跑完，结果可用（下条消息命中）。"""
        finished = []

        def _mk(url, **kw):
            async def _run():
                await asyncio.sleep(0.15)
                finished.append(url)
                return f"描述:{url}"
            return asyncio.create_task(_run())
        monkeypatch.setattr(vision, "describe_image_shared", _mk)

        out = await vision.describe_images(["http://x/1.png"], model=object(), wait=0.03)
        assert out == {"http://x/1.png": "[图片]"}  # 本次降级
        await asyncio.sleep(0.3)                     # 但任务没被杀
        assert finished == ["http://x/1.png"]

    @pytest.mark.asyncio
    async def test_wait_zero_means_wait_all(self, monkeypatch):
        monkeypatch.setattr(vision, "describe_image_shared", _mk_task_factory(0.1))
        out = await vision.describe_images(["http://x/1.png"], model=object(), wait=0)
        assert out == {"http://x/1.png": "描述:http://x/1.png"}

    @pytest.mark.asyncio
    async def test_mixed_fast_and_slow(self, monkeypatch):
        """多张图：完成的用真实描述，没完成的占位——互不影响。"""
        def _mk(url, **kw):
            async def _run():
                if "slow" in url:
                    await asyncio.sleep(5.0)
                return f"描述:{url}"
            return asyncio.create_task(_run())
        monkeypatch.setattr(vision, "describe_image_shared", _mk)

        out = await vision.describe_images(
            ["http://x/fast.png", "http://x/slow.png"], model=object(), wait=0.05)
        assert out["http://x/fast.png"] == "描述:http://x/fast.png"
        assert out["http://x/slow.png"] == "[图片]"
        await asyncio.sleep(0)
        for t in list(getattr(vision, "_PENDING", {}).values()):
            t.cancel()

    @pytest.mark.asyncio
    async def test_stickers_bounded_and_placeholder(self, monkeypatch):
        monkeypatch.setattr(vision, "describe_image_shared", _mk_task_factory(5.0))
        out = await vision.describe_stickers(["http://x/s.png"], model=object(), wait=0.05)
        assert out == {"http://x/s.png": "[表情]"}
        await asyncio.sleep(0)
        for t in list(getattr(vision, "_PENDING", {}).values()):
            t.cancel()

    @pytest.mark.asyncio
    async def test_exception_task_degrades(self, monkeypatch):
        """任务本身抛异常：降级占位而不是炸掉决策。"""
        def _mk(url, **kw):
            async def _run():
                raise RuntimeError("vlm down")
            return asyncio.create_task(_run())
        monkeypatch.setattr(vision, "describe_image_shared", _mk)
        out = await vision.describe_images(["http://x/1.png"], model=object(), wait=0.5)
        assert out == {"http://x/1.png": "[图片]"}

    @pytest.mark.asyncio
    async def test_empty_urls(self):
        assert await vision.describe_images([], model=object()) == {}
        assert await vision.describe_stickers([], model=object()) == {}


class TestPerceptionWaitConfig:
    def test_default_wait(self, monkeypatch):
        """无配置时默认 3 秒。"""
        class _Cfg:
            raw = {}
        monkeypatch.setattr("junjun_core.config.get_global_config", lambda: _Cfg())
        assert vision._perception_wait() == 3.0

    def test_config_override(self, monkeypatch):
        class _Cfg:
            raw = {"perception": {"ready_wait_seconds": 1.5}}
        monkeypatch.setattr("junjun_core.config.get_global_config", lambda: _Cfg())
        assert vision._perception_wait() == 1.5

    def test_config_broken_degrades(self, monkeypatch):
        monkeypatch.setattr(
            "junjun_core.config.get_global_config",
            lambda: (_ for _ in ()).throw(RuntimeError("no config")))
        assert vision._perception_wait() == 3.0
