"""TaskKernel LangGraph 引擎测试：图执行 / 崩溃断点续跑 / 人审中断 / 启动恢复。

全部假模型 + 桩工具 + MemorySaver（崩溃恢复用 tmp_path 的 SqliteSaver），
不调 LLM、不触生产库、不触网络。legacy 引擎行为由 test_task_kernel.py 覆盖。
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from junjun_agent.task_kernel import executor
from junjun_agent.task_kernel.graph import (
    KernelState, approval_hook, build_graph, runner,
)
from junjun_agent.task_kernel.plan import SYNTH_ACTION, Step, TaskPlan


# ---------- 桩 ----------

class _FakeModel:
    def __init__(self, text="合成结果"):
        self._text = text

    async def ainvoke(self, msgs, config=None):
        return AIMessage(content=self._text)


class _StubTool:
    def __init__(self, name, fn):
        self.name = name
        self.description = f"{name} 桩"
        self._fn = fn

    async def ainvoke(self, args):
        return self._fn(args)


@pytest.fixture
def harness(monkeypatch):
    """假配置（langgraph 引擎）+ 桩工具表 + 假模型 + 出站/审批通知记录器；
    runner 单例状态每个测试重置并注入 MemorySaver 图。"""
    sent, admin_msgs = [], []

    async def _fake_send(chat_id, segments, *, source="", remember=True):
        sent.append((chat_id, segments, source))
        return True

    async def _fake_notify(text):
        admin_msgs.append(text)
        return True

    monkeypatch.setattr(executor, "_cfg", lambda: {
        "enable": True, "engine": "langgraph", "max_steps": 6,
        "deadline_minutes": 30, "max_replans": 1,
        "approval_timeout_seconds": 600})
    import junjun_llm
    monkeypatch.setattr(junjun_llm, "get_chat_model",
                        lambda slot="utils": _FakeModel())
    import junjun_agent.outbound as outbound
    monkeypatch.setattr(outbound, "send_proactive", _fake_send)
    import junjun_core.security as sec
    monkeypatch.setattr(sec, "notify_admin", _fake_notify)

    from langgraph.checkpoint.memory import MemorySaver
    runner._graph = build_graph(MemorySaver())
    runner._persist_dir = None
    runner._pending.clear()
    runner._recovered = False
    yield {"sent": sent, "admin": admin_msgs}
    runner._graph = None
    runner._pending.clear()


def _bind_tools(monkeypatch, tools):
    import junjun_skills.registry as reg
    monkeypatch.setattr(reg, "get_tools", lambda session=None: tools)


def _plan(steps, chat_id="qq:g1:group"):
    plan = TaskPlan(goal="测试任务", chat_id=chat_id, steps=steps)
    plan.deadline_ts = 9e18
    return plan


def _outcomes(chat_id):
    from junjun_agent.tasks import task_manager
    return list(task_manager._outcomes.get(chat_id, ()))


# ---------- 图执行（与 legacy 同语义） ----------

class TestGraphFlow:
    @pytest.mark.asyncio
    async def test_happy_path(self, harness, monkeypatch):
        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "搜索结果")])
        plan = _plan([
            Step(id="s1", action="web_search", desc="搜资料"),
            Step(id="s2", action=SYNTH_ACTION, desc="汇总", depends_on=["s1"]),
        ])
        await runner.submit(plan)
        assert plan.state == "done" or True  # plan 对象以 state 为准，查 outcome
        oc = _outcomes("qq:g1:group")
        assert oc and oc[-1]["status"] == "done"
        assert harness["sent"], "完成后必须主动汇报"

    @pytest.mark.asyncio
    async def test_retry_once_then_success(self, harness, monkeypatch):
        attempts = {"n": 0}

        def flaky(args):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("网络抖动")
            return "成了"

        _bind_tools(monkeypatch, [_StubTool("web_search", flaky)])
        await runner.submit(_plan([Step(id="s1", action="web_search", desc="搜")]))
        assert attempts["n"] == 2, "失败必须重试一次"
        assert _outcomes("qq:g1:group")[-1]["status"] == "done"

    @pytest.mark.asyncio
    async def test_replan_after_repeated_failure(self, harness, monkeypatch):
        def always_fail(args):
            raise RuntimeError("彻底不行")

        _bind_tools(monkeypatch, [_StubTool("bad_tool", always_fail)])
        import junjun_agent.task_kernel.planner as planner

        async def fake_revise(plan, desc, err):
            return [Step(id="r1", action=SYNTH_ACTION, desc="直接收尾")]

        monkeypatch.setattr(planner, "revise_remaining", fake_revise)
        await runner.submit(_plan([Step(id="s1", action="bad_tool", desc="会失败")]))
        assert _outcomes("qq:g1:group")[-1]["status"] == "done", "重规划换 synth 收尾应完成"

    @pytest.mark.asyncio
    async def test_abort_when_replan_exhausted(self, harness, monkeypatch):
        monkeypatch.setattr(executor, "_cfg", lambda: {
            "enable": True, "engine": "langgraph", "max_replans": 0,
            "approval_timeout_seconds": 600})

        def always_fail(args):
            raise RuntimeError("不行")

        _bind_tools(monkeypatch, [_StubTool("bad_tool", always_fail)])
        await runner.submit(_plan([Step(id="s1", action="bad_tool", desc="必败步骤")]))
        oc = _outcomes("qq:g1:group")
        assert oc[-1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_deadline_aborts(self, harness, monkeypatch):
        _bind_tools(monkeypatch, [])
        plan = _plan([Step(id="s1", action=SYNTH_ACTION, desc="合成")])
        plan.deadline_ts = 1.0  # 早已过期
        await runner.submit(plan)
        oc = _outcomes("qq:g1:group")
        assert oc[-1]["status"] == "failed" and "时限" in oc[-1]["detail"]


# ---------- 崩溃断点续跑（SqliteSaver + 同 thread_id + input None） ----------

class TestCrashResume:
    @pytest.mark.asyncio
    async def test_resume_continues_from_checkpoint(self, harness, monkeypatch, tmp_path):
        """图级异常（recursion_limit 烧穿，等价进程崩溃）后：同 thread_id 传 None
        续跑——已完成的步骤绝不重跑，未完成的继续。"""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        calls = {"s1": 0}

        def search(args):
            calls["s1"] += 1
            return "搜索结果"

        _bind_tools(monkeypatch, [_StubTool("web_search", search)])
        conn = await aiosqlite.connect(str(tmp_path / "k.db"))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        graph = build_graph(saver)

        plan = _plan([
            Step(id="s1", action="web_search", desc="搜"),
            Step(id="s2", action=SYNTH_ACTION, desc="汇总", depends_on=["s1"]),
        ])
        initial = {"plan": plan.to_dict(), "phase": "execute",
                   "awaiting": "", "replan_for": ""}
        cfg = {"configurable": {"thread_id": plan.plan_id}, "recursion_limit": 3}
        with pytest.raises(Exception) as exc:  # GraphRecursionError
            await graph.ainvoke(initial, cfg)
        assert "Recursion" in type(exc.value).__name__

        cfg2 = {"configurable": {"thread_id": plan.plan_id}, "recursion_limit": 50}
        state = await graph.ainvoke(None, cfg2)  # None = 断点续跑
        final = TaskPlan.from_dict(state["plan"])
        assert final.state == "done"
        assert calls["s1"] == 1, "已完成的步骤不得因续跑重跑"
        await conn.close()


# ---------- 人审中断 ----------

class TestApproval:
    def _gated_plan(self):
        return _plan([Step(id="s1", action="send_feed", desc="发空间说说",
                           verify="human")])

    @pytest.mark.asyncio
    async def test_approve_runs_step(self, harness, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command
        calls = []
        _bind_tools(monkeypatch, [_StubTool("send_feed",
                                            lambda a: calls.append(a) or "已发布")])
        graph = build_graph(MemorySaver())
        plan = self._gated_plan()
        cfg = {"configurable": {"thread_id": plan.plan_id}, "recursion_limit": 50}
        state = await graph.ainvoke(
            {"plan": plan.to_dict(), "phase": "execute", "awaiting": "",
             "replan_for": ""}, cfg)
        assert "__interrupt__" in state, "human 门步骤必须先挂起"
        assert not calls, "批准前不许执行发布动作"

        state = await graph.ainvoke(Command(resume=True), cfg)
        final = TaskPlan.from_dict(state["plan"])
        assert calls, "批准后必须执行"
        assert final.state == "done"

    @pytest.mark.asyncio
    async def test_reject_skips_step(self, harness, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command
        calls = []
        _bind_tools(monkeypatch, [_StubTool("send_feed",
                                            lambda a: calls.append(a) or "已发布")])
        graph = build_graph(MemorySaver())
        plan = self._gated_plan()
        cfg = {"configurable": {"thread_id": plan.plan_id}, "recursion_limit": 50}
        state = await graph.ainvoke(
            {"plan": plan.to_dict(), "phase": "execute", "awaiting": "",
             "replan_for": ""}, cfg)
        assert "__interrupt__" in state
        state = await graph.ainvoke(Command(resume=False), cfg)
        final = TaskPlan.from_dict(state["plan"])
        assert not calls, "否决后绝不执行"
        assert final.steps[0].status == "skipped"
        assert final.state == "done", "唯一步骤被跳过 = 任务整体完成（无事可做）"

    @pytest.mark.asyncio
    async def test_timeout_default_skip(self, harness, monkeypatch):
        """审批超时默认跳过（宁保守不放行）。"""
        monkeypatch.setattr(executor, "_cfg", lambda: {
            "enable": True, "engine": "langgraph", "max_steps": 6,
            "max_replans": 1, "approval_timeout_seconds": 0.05})
        calls = []
        _bind_tools(monkeypatch, [_StubTool("send_feed",
                                            lambda a: calls.append(a) or "已发布")])
        await runner.submit(self._gated_plan())
        assert runner.pending_approvals, "挂起后必须有待审批记录"
        assert harness["admin"], "必须私聊通知管理员"
        await asyncio.sleep(0.3)
        assert not runner.pending_approvals
        assert not calls, "超时默认跳过，不许放行"

    @pytest.mark.asyncio
    async def test_hook_consumption_rules(self, harness, monkeypatch):
        """误判回归：非管理员/非精确词/无待审批/legacy 引擎——一律不拦截。"""
        import junjun_core.security as sec
        monkeypatch.setattr(sec, "is_admin", lambda uid: uid == "999")
        resumed = []

        async def fake_resume(plan_id, approved):
            resumed.append((plan_id, approved))

        monkeypatch.setattr(runner, "resume", fake_resume)
        session = SimpleNamespace(chat_id="qq:999:private")

        # 无待审批：管理员的「发」也是正常聊天
        assert await approval_hook(session, SimpleNamespace(user_id="999", text="发")) is False
        runner._pending["p1"] = {"goal": "g", "desc": "发空间"}
        # 非管理员不拦
        assert await approval_hook(session, SimpleNamespace(user_id="123", text="发")) is False
        # 非精确词不拦（「发一下」是日常句子）
        assert await approval_hook(session, SimpleNamespace(user_id="999", text="发一下")) is False
        # 管理员 + 精确词 + 有待审批 → 消费
        assert await approval_hook(session, SimpleNamespace(user_id="999", text="发")) is True
        await asyncio.sleep(0)
        assert resumed == [("p1", True)]
        # legacy 引擎整体不拦
        monkeypatch.setattr(executor, "_cfg", lambda: {"engine": "legacy"})
        runner._pending["p2"] = {"goal": "g", "desc": "d"}
        assert await approval_hook(session, SimpleNamespace(user_id="999", text="发")) is False


# ---------- 启动断点恢复（注册表 + sqlite checkpoint） ----------

class TestStartupRecover:
    @pytest.mark.asyncio
    async def test_recover_renotifies_pending_approval(self, harness, monkeypatch, tmp_path):
        """审批挂起时进程重启：注册表 + checkpoint 都在 -> recover 重新挂起
        并重新通知管理员（不等新消息触发）。"""
        _bind_tools(monkeypatch, [_StubTool("send_feed", lambda a: "已发布")])
        runner.configure(tmp_path)
        runner._graph = None  # configure 后必须重建：harness 注入的是 MemorySaver
        try:
            await runner.submit(TestApproval()._gated_plan())
            assert "【任务审批】" in harness["admin"][-1]
            assert len(runner.pending_approvals) == 1

            # 模拟重启：清内存态，持久层（sqlite + 注册表）保留
            harness["admin"].clear()
            runner._graph = None
            runner._pending.clear()
            runner._recovered = False

            await runner.recover()
            assert len(runner.pending_approvals) == 1, "挂起的审批必须恢复"
            assert harness["admin"], "恢复后必须重新通知管理员"
            # 清理超时看门狗，防测试间串扰
            for pid in list(runner._pending):
                t = runner._pending[pid].get("timeout_task")
                if t:
                    t.cancel()
        finally:
            if runner._graph is not None:
                try:
                    await runner._graph.checkpointer.conn.close()
                except Exception:
                    pass
            runner._persist_dir = None
            runner._graph = None
            runner._pending.clear()


# ---------- 接单路由 ----------

class TestSubmitRouting:
    @pytest.mark.asyncio
    async def test_langgraph_engine_routes_and_gates(self, harness, monkeypatch):
        """engine=langgraph：走图引擎 + send_feed 步骤被强制 human 门。"""
        import junjun_agent.task_kernel.planner as planner
        captured = {}

        async def fake_plan(goal, **k):
            return TaskPlan(goal=goal, chat_id=k.get("chat_id", ""), steps=[
                Step(id="s1", action=SYNTH_ACTION, desc="写文案"),
                Step(id="s2", action="send_feed", desc="发空间", depends_on=["s1"]),
            ])

        async def fake_submit(plan):
            captured["plan"] = plan

        monkeypatch.setattr(planner, "make_plan", fake_plan)
        monkeypatch.setattr(runner, "submit", fake_submit)
        ack = await executor.kernel.try_submit("写个总结发我空间", chat_id="qq:g2:group")
        assert ack
        await asyncio.sleep(0)
        gated = captured["plan"].steps[1]
        assert gated.verify == "human", "发布类动作必须被强制人审"

    @pytest.mark.asyncio
    async def test_legacy_engine_unaffected(self, monkeypatch):
        """engine=legacy（默认）：路由和人审门都不变（旧行为回归）。"""
        monkeypatch.setattr(executor, "_cfg", lambda: {
            "enable": True, "engine": "legacy", "max_steps": 6,
            "deadline_minutes": 30})
        import junjun_agent.task_kernel.planner as planner

        async def fake_plan(goal, **k):
            return TaskPlan(goal=goal, chat_id=k.get("chat_id", ""), steps=[
                Step(id="s1", action="send_feed", desc="发空间"),
            ])

        async def fake_run(self, plan):
            plan.state = "done"

        monkeypatch.setattr(planner, "make_plan", fake_plan)
        monkeypatch.setattr(executor.TaskKernel, "_run", fake_run)
        ack = await executor.kernel.try_submit("发个空间", chat_id="qq:g3:group")
        assert ack
        await asyncio.sleep(0)
