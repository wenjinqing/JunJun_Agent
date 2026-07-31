"""异步任务队列测试：接单/认领/完成/失败/超时/取消/sweep 恢复/汇报截断。

DB 用内存库隔离；gateway 与 LLM 全部打桩；handler 用假执行器。
"""

import asyncio
import time

import pytest
from peewee import SqliteDatabase

import junjun_core.config.config as cfg_mod
from junjun_core.database import models as m

test_db = SqliteDatabase(":memory:")

CHAT = "qq:12345:group"

# env fixture 会把 _notify 换成记录器；TestNotify 需要真身，提前抓引用
from junjun_agent.loop import async_jobs as _aj_mod
_REAL_NOTIFY = _aj_mod._notify


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    _set_config({"async_task": {"enable": True, "max_concurrent": 2,
                                "job_timeout_seconds": 600, "max_pending_per_chat": 2,
                                "report_max_chars": 50, "subagent_max_iter": 4}})
    with test_db.bind_ctx([m.AsyncJob]):
        test_db.create_tables([m.AsyncJob])
        m.AsyncJob.delete().execute()
        from junjun_agent.loop import async_jobs
        async_jobs._HANDLERS.pop("test", None)
        notified = []

        async def _fake_notify(job):
            notified.append((job.job_id, job.status))
        monkeypatch.setattr(async_jobs, "_notify", _fake_notify)
        yield async_jobs, notified
        async_jobs._HANDLERS.pop("test", None)
    cfg_mod.global_config = old


def _submit(aj, title="调研一下xxx", chat_id=CHAT, kind="test"):
    job, err = aj.submit_job(kind, title, {"task": title}, chat_id,
                             "111", "甲", kick=False)
    assert err == ""
    return job


class TestSubmit:
    def test_submit_ok(self, env):
        aj, _ = env
        aj._HANDLERS["test"] = lambda j, p: None
        job = _submit(aj)
        assert job.status == "pending" and len(job.job_id) == 10

    def test_unknown_kind_rejected(self, env):
        aj, _ = env
        job, err = aj.submit_job("nope", "t", {}, CHAT, kick=False)
        assert job is None and "执行器" in err

    def test_per_chat_cap(self, env):
        aj, _ = env
        aj._HANDLERS["test"] = lambda j, p: None
        _submit(aj, "t1")
        _submit(aj, "t2")
        job, err = aj.submit_job("test", "t3", {}, CHAT, kick=False)
        assert job is None and "上限" in err  # max_pending_per_chat=2


class TestRun:
    @pytest.mark.asyncio
    async def test_success(self, env):
        aj, notified = env

        async def h(job, payload):
            return "结果正文"
        aj._HANDLERS["test"] = h
        job = _submit(aj)
        await aj._run(job.job_id)
        row = m.AsyncJob.get_by_id(job.id)
        assert row.status == "done" and row.result == "结果正文"
        assert row.attempts == 1 and notified == [(job.job_id, "done")]

    @pytest.mark.asyncio
    async def test_handler_raises(self, env):
        aj, notified = env

        async def h(job, payload):
            raise ValueError("炸了")
        aj._HANDLERS["test"] = h
        job = _submit(aj)
        await aj._run(job.job_id)
        row = m.AsyncJob.get_by_id(job.id)
        assert row.status == "failed" and "炸了" in row.error
        assert notified == [(job.job_id, "failed")]

    @pytest.mark.asyncio
    async def test_timeout(self, env, monkeypatch):
        aj, notified = env
        monkeypatch.setitem(cfg_mod.global_config.raw["async_task"],
                            "job_timeout_seconds", 0.05)

        async def h(job, payload):
            await asyncio.sleep(5)
        aj._HANDLERS["test"] = h
        job = _submit(aj)
        await aj._run(job.job_id)
        row = m.AsyncJob.get_by_id(job.id)
        assert row.status == "failed" and "超时" in row.error

    @pytest.mark.asyncio
    async def test_double_run_claims_once(self, env):
        """immediate kick 与 sweep 并发认领：handler 只跑一次。"""
        aj, _ = env
        calls = []

        async def h(job, payload):
            calls.append(1)
            await asyncio.sleep(0.05)
            return "ok"
        aj._HANDLERS["test"] = h
        job = _submit(aj)
        await asyncio.gather(aj._run(job.job_id), aj._run(job.job_id))
        row = m.AsyncJob.get_by_id(job.id)
        assert len(calls) == 1 and row.attempts == 1 and row.status == "done"


class TestCancel:
    def test_cancel_pending(self, env):
        aj, _ = env
        aj._HANDLERS["test"] = lambda j, p: None
        job = _submit(aj)
        msg = aj.cancel_job(job.job_id, "111")
        assert "已取消" in msg
        assert m.AsyncJob.get_by_id(job.id).status == "cancelled"

    def test_cancel_permission(self, env):
        aj, _ = env
        aj._HANDLERS["test"] = lambda j, p: None
        job = _submit(aj)
        msg = aj.cancel_job(job.job_id, "999")  # 非本人非管理员
        assert "只有本人或管理员" in msg
        assert m.AsyncJob.get_by_id(job.id).status == "pending"

    @pytest.mark.asyncio
    async def test_cancel_running(self, env):
        aj, notified = env
        started = asyncio.Event()

        async def h(job, payload):
            started.set()
            await asyncio.sleep(60)
        aj._HANDLERS["test"] = h
        job = _submit(aj)
        task = asyncio.create_task(aj._run(job.job_id))
        aj._running[job.job_id] = task  # 模拟 _kick 的登记（cancel 靠它找到协程）
        await asyncio.wait_for(started.wait(), 2)
        aj.cancel_job(job.job_id, "111")
        await asyncio.gather(task, return_exceptions=True)
        row = m.AsyncJob.get_by_id(job.id)
        assert row.status == "cancelled" and notified == []  # 取消不打扰


class TestSweep:
    @pytest.mark.asyncio
    async def test_stuck_recovery_and_pending_kick(self, env, monkeypatch):
        aj, notified = env
        kicked = []
        monkeypatch.setattr(aj, "_kick", lambda jid: kicked.append(jid))
        old = time.time() - 99999
        # 崩溃残留：attempts=1 -> 回炉 pending
        m.AsyncJob.create(job_id="stuck1", kind="test", title="a", status="running",
                          started_at=old, attempts=1, chat_id=CHAT)
        # 崩溃残留：attempts=2 -> 重试超限判死
        m.AsyncJob.create(job_id="stuck2", kind="test", title="b", status="running",
                          started_at=old, attempts=2, chat_id=CHAT)
        # 重启遗留 pending -> 补捞
        m.AsyncJob.create(job_id="pend1", kind="test", title="c", status="pending",
                          chat_id=CHAT)
        await aj.sweep_jobs()
        assert m.AsyncJob.get(m.AsyncJob.job_id == "stuck1").status == "pending"
        dead = m.AsyncJob.get(m.AsyncJob.job_id == "stuck2")
        assert dead.status == "failed" and "中断" in dead.error
        assert set(kicked) == {"stuck1", "pend1"}
        assert ("stuck2", "failed") in notified

    @pytest.mark.asyncio
    async def test_retention_cleanup(self, env, monkeypatch):
        aj, _ = env
        monkeypatch.setattr(aj, "_kick", lambda jid: None)
        old = time.time() - 8 * 86400
        m.AsyncJob.create(job_id="old1", kind="test", title="x", status="done",
                          finished_at=old, chat_id=CHAT)
        m.AsyncJob.create(job_id="new1", kind="test", title="y", status="done",
                          finished_at=time.time(), chat_id=CHAT)
        await aj.sweep_jobs()
        assert m.AsyncJob.get_or_none(m.AsyncJob.job_id == "old1") is None
        assert m.AsyncJob.get_or_none(m.AsyncJob.job_id == "new1") is not None


class TestNotify:
    @pytest.mark.asyncio
    async def test_report_format_and_truncation(self, env, monkeypatch):
        """汇报 = 人设开场白 + 保真正文；超长截断带提示；群路由正确。"""
        aj, _ = env
        monkeypatch.setattr(aj, "_notify", _REAL_NOTIFY)  # 还原被 fixture 换掉的真身
        sent = []

        class _FakeGW:
            async def send_reply(self, rs):
                sent.append(rs)
        import junjun_core.gateway.router as router_mod
        monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())

        async def _lead(job, ok):
            return "做好啦："
        monkeypatch.setattr(aj, "_persona_lead", _lead)

        job = m.AsyncJob.create(job_id="n1", kind="test", title="调研",
                                status="done", result="正" * 100,
                                chat_id=CHAT, user_id="111")
        await aj._notify(job)
        assert len(sent) == 1
        rs = sent[0]
        assert rs.target_group_id == "12345" and rs.target_user_id is None
        text = rs.segments[0].data
        assert text.startswith("做好啦：") and "正" * 50 in text
        assert "太长" in text  # report_max_chars=50 截断提示


class TestAgentTaskHandler:
    @pytest.mark.asyncio
    async def test_subagent_returns_content(self, env):
        """隔离子 agent：假模型直出正文 -> handler 返回正文。"""
        from langchain_core.messages import AIMessage
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel)

        class _BindableFake(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        from junjun_skills.plugins.async_task import tools as plugin
        fake = _BindableFake(responses=[AIMessage(content="调研报告正文")])
        job = type("J", (), {"chat_id": CHAT, "title": "t", "kind": "agent_task"})()
        out = await plugin._agent_task_handler(job, {"task": "查xxx"}, model=fake)
        assert out == "调研报告正文"


class TestTool:
    def test_run_background_task(self, env):
        """LLM 工具入口：contextvar 路由 -> 落表 pending。"""
        from junjun_skills.plugins.async_task import tools as plugin
        from junjun_skills.builtin.memory_skills import current_chat_id
        from junjun_core.security import current_user_id, current_nickname
        plugin.async_jobs._HANDLERS.setdefault("agent_task", lambda j, p: None)
        t1 = current_chat_id.set(CHAT)
        t2 = current_user_id.set("111")
        t3 = current_nickname.set("甲")
        try:
            out = plugin.run_background_task.invoke({"task": "帮我调研一下xxx"})
            assert "接单成功" in out
            row = m.AsyncJob.get()
            assert row.kind == "agent_task" and row.status == "pending"
            assert row.chat_id == CHAT and row.user_id == "111"
            assert "查xxx" in row.payload or "调研一下xxx" in row.payload
        finally:
            current_chat_id.reset(t1)
            current_user_id.reset(t2)
            current_nickname.reset(t3)

    def test_list_empty(self, env):
        aj, _ = env
        assert "还没有后台任务" in aj.list_for_chat(CHAT)
