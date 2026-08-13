"""fire_and_forget 强引用留存测试（2026-08-13 生产实锤：裸 create_task
的后台规划任务被 GC 收走，跑了 8 分钟后人间蒸发——无日志无汇报）。

核心断言：任务在途时模块级集合持有强引用（GC 收不走），完成后释放；
异常/取消路径不炸事件循环且有日志。
"""

import asyncio
import gc

import pytest

from junjun_core import bg_tasks


class TestFireAndForget:
    @pytest.mark.asyncio
    async def test_held_while_pending_and_released_after(self):
        """在途期间强引用存在（GC 收不走），完成后自动释放。"""
        started = asyncio.Event()
        release = asyncio.Event()

        async def job():
            started.set()
            await release.wait()
            return 42

        task = bg_tasks.fire_and_forget(job(), name="t-hold")
        await started.wait()
        before = bg_tasks.pending_count()
        assert any(t is task for t in bg_tasks._BG_TASKS)
        gc.collect()  # 在途 GC：任务必须存活
        assert not task.done()
        release.set()
        assert await task == 42
        await asyncio.sleep(0)  # 让 done 回调跑完
        assert bg_tasks.pending_count() == before - 1

    @pytest.mark.asyncio
    async def test_survives_without_caller_reference(self):
        """调用方不存引用（唯一防线是模块集合）：任务照样跑完。"""
        done = asyncio.Event()

        async def job():
            await asyncio.sleep(0.01)
            done.set()

        bg_tasks.fire_and_forget(job(), name="t-noref")  # 返回值刻意丢弃
        gc.collect()
        assert await asyncio.wait_for(done.wait(), timeout=2)

    @pytest.mark.asyncio
    async def test_exception_does_not_crash_loop_and_cleans_up(self):
        """异常逃逸：done 回调吃掉（打日志），集合清理，不污染事件循环。"""

        async def boom():
            raise ValueError("炸了")

        task = bg_tasks.fire_and_forget(boom(), name="t-boom")
        with pytest.raises(ValueError):
            await task
        await asyncio.sleep(0)
        assert not any(t is task for t in bg_tasks._BG_TASKS)

    @pytest.mark.asyncio
    async def test_cancelled_path_clean(self):
        """主动取消：回调走 cancelled 分支，集合清理。"""

        async def forever():
            await asyncio.sleep(60)

        task = bg_tasks.fire_and_forget(forever(), name="t-cancel")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert not any(t is task for t in bg_tasks._BG_TASKS)
