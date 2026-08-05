"""调度器启动错峰测试：interval 任务首轮不同时到期，防 LLM 洪峰。"""

import time

from junjun_agent.loop.scheduler import ScheduledTask, Scheduler


class TestStartStagger:
    def test_interval_tasks_staggered(self, monkeypatch):
        import asyncio

        s = Scheduler()
        for i in range(5):
            s.add(ScheduledTask(f"t{i}", _noop, interval=60))
        # 不真正起 loop（需要运行中的事件循环），只验证 start 的错峰副作用
        created = []

        def _fake_create_task(coro, name=""):
            created.append(name)
            coro.close()  # 不跑 loop，直接关掉防告警

            class _T:
                def done(self): return True

            return _T()

        monkeypatch.setattr(asyncio, "create_task", _fake_create_task)
        before = time.time()
        s.start()
        now = time.time()
        tasks = [s._tasks[f"t{i}"] for i in range(5)]
        # 第 1 个立即到期，后续按 20s 错开 -> 此刻不到期
        assert tasks[0].due(now)
        for t in tasks[1:]:
            assert not t.due(now)
        # 错峰上限：不超过 interval 的一半
        for t in tasks:
            assert (t._last_run - (before - t.interval)) <= t.interval * 0.5 + 1

    def test_cron_tasks_untouched(self, monkeypatch):
        import asyncio
        s = Scheduler()
        s.add(ScheduledTask("cron1", _noop, cron_hour=8, cron_minute=0))
        monkeypatch.setattr(asyncio, "create_task", _fake_create_task())
        s.start()
        assert s._tasks["cron1"]._last_run == 0.0  # cron 不参与错峰


def _fake_create_task():
    def _f(coro, name=""):
        coro.close()

        class _T:
            def done(self): return True
        return _T()
    return _f


async def _noop():
    pass
