"""任务有下文（2026-08-04「图呢」事件）：失败自动重试 / 结局登记 /
决策注入 / 短期记忆回写 / list_background_tasks 合并成品任务。"""

import asyncio

import pytest

from junjun_agent.tasks import TaskManager, task_manager
from junjun_core.contracts import ReplySegment


@pytest.fixture
def tm():
    m = TaskManager()
    sent = []

    async def _send(chat_id, segments):
        sent.append(segments)
        return True
    m._send = _send
    yield m, sent


class TestAutoRetry:
    @pytest.mark.asyncio
    async def test_retry_once_on_failure(self, tm, monkeypatch):
        """首次失败自动重试一次；第二次成功则直发成品。"""
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: True))
        calls = []

        async def work():
            calls.append(1)
            if len(calls) == 1:
                return None                      # 第一次无产出
            return [ReplySegment(type="image", data="http://x/a.png")]

        ack = await m.submit(kind="ai_draw", work=work, done_text="画好了",
                             fail_text="画砸了", timeout=5, chat_id="qq:1:private")
        assert ack
        await asyncio.sleep(3.5)                 # 重试间隔 3s
        assert len(calls) == 2
        assert sent and sent[0][0].data == "画好了"
        out = m._outcomes["qq:1:private"][-1]
        assert out["status"] == "done"

    @pytest.mark.asyncio
    async def test_retry_disabled_config(self, tm, monkeypatch):
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        calls = []

        async def work():
            calls.append(1)
            return None

        await m.submit(kind="ai_draw", work=work, fail_text="画砸了",
                       timeout=5, chat_id="qq:1:private")
        await asyncio.sleep(0.2)
        assert len(calls) == 1                   # 不重试
        assert sent[0][0].data == "画砸了"
        assert m._outcomes["qq:1:private"][-1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_both_attempts_fail_records_failure(self, tm, monkeypatch):
        m, sent = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: True))

        async def work():
            raise RuntimeError("api down")

        await m.submit(kind="ai_draw", work=work, fail_text="画砸了",
                       timeout=5, chat_id="qq:1:private")
        await asyncio.sleep(3.5)
        out = m._outcomes["qq:1:private"][-1]
        assert out["status"] == "failed" and "RuntimeError" in out["detail"]
        assert sent[0][0].data == "画砸了"


class TestOutcomeVisibility:
    @pytest.mark.asyncio
    async def test_status_block_running_and_done(self, tm, monkeypatch):
        """在途与结局都进决策注入块。"""
        m, _ = tm
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        ev = asyncio.Event()

        async def work_slow():
            await ev.wait()
            return [ReplySegment(type="text", data="x")]

        await m.submit(kind="ai_draw", work=work_slow, timeout=30,
                       chat_id="qq:1:private")
        block = m.task_status_block("qq:1:private")
        assert "画图" in block and "进行中" in block
        ev.set()
        await asyncio.sleep(0.2)
        block = m.task_status_block("qq:1:private")
        assert "完成" in block
        # 别的会话不受影响
        assert m.task_status_block("qq:2:private") == ""

    def test_list_for_chat_empty(self, tm):
        m, _ = tm
        assert m.list_for_chat("qq:9:private") == ""

    @pytest.mark.asyncio
    async def test_list_background_tasks_merges_task_manager(self, tm, monkeypatch):
        """工具合并：成品任务（task_manager）+ 派活任务（async_jobs）都可见。"""
        m, _ = tm
        import junjun_skills.plugins.async_task.tools as att
        monkeypatch.setattr(att, "async_jobs",
                            type("AJ", (), {"list_for_chat": staticmethod(
                                lambda cid: "这个会话还没有后台任务。")}))
        from junjun_agent import tasks as tasks_mod
        monkeypatch.setattr(tasks_mod, "task_manager", m)
        monkeypatch.setattr(TaskManager, "_auto_retry", staticmethod(lambda: False))
        m._outcomes.setdefault("qq:1:private", __import__("collections").deque(maxlen=10)).append(
            {"ts": __import__("time").time(), "kind": "ai_draw",
             "status": "failed", "detail": "超时"})
        from junjun_skills.builtin import memory_skills
        token = memory_skills.current_chat_id.set("qq:1:private")
        try:
            out = att.list_background_tasks.invoke({})
        finally:
            memory_skills.current_chat_id.reset(token)
        assert "成品任务" in out and "画图" in out and "失败" in out
        assert "还没有后台任务" in out           # async_jobs 段也在
