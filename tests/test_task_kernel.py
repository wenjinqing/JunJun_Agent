"""TaskKernel 离线测试：步骤图解析防御 + 执行循环（retry/replan/abort/汇报）。

全部用假模型与桩工具，不调 LLM、不触 DB、不触网络。
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from junjun_agent.task_kernel import executor
from junjun_agent.task_kernel.plan import SYNTH_ACTION, Step, TaskPlan, parse_plan


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
    """接线：假配置（enable）、桩工具表、假模型、出站记录器。"""
    sent = []

    async def _fake_send(chat_id, segments, *, source="", remember=True):
        sent.append((chat_id, segments, source))
        return True

    monkeypatch.setattr(executor, "_cfg",
                        lambda: {"enable": True, "max_steps": 6,
                                 "deadline_minutes": 30, "max_replans": 1})
    import junjun_llm
    monkeypatch.setattr(junjun_llm, "get_chat_model",
                        lambda slot="utils": _FakeModel())
    import junjun_agent.outbound as outbound
    monkeypatch.setattr(outbound, "send_proactive", _fake_send)
    return {"sent": sent}


def _bind_tools(monkeypatch, tools):
    import junjun_skills.registry as reg
    monkeypatch.setattr(reg, "get_tools", lambda session=None: tools)


def _outcomes(chat_id):
    from junjun_agent.tasks import task_manager
    return list(task_manager._outcomes.get(chat_id, ()))


# ---------- parse_plan 防御 ----------

class TestParsePlan:
    def test_drops_unknown_tools(self):
        p = parse_plan({"steps": [
            {"id": "s1", "action": "编出来的工具", "desc": "x"},
            {"id": "s2", "action": "web_search", "desc": "搜"},
        ]}, goal="g", chat_id="c", user_id="u", valid_actions={"web_search"})
        assert [s.id for s in p.steps] == ["s2"]

    def test_all_invalid_returns_none(self):
        assert parse_plan({"steps": [{"action": "不存在", "desc": "x"}]},
                          goal="g", chat_id="c", user_id="u",
                          valid_actions=set()) is None
        assert parse_plan({"steps": []}, goal="g", chat_id="c",
                          user_id="u", valid_actions=set()) is None

    def test_forward_deps_only(self):
        p = parse_plan({"steps": [
            {"id": "s1", "action": "a", "desc": "1", "depends_on": ["s2"]},  # 后向依赖，丢
            {"id": "s2", "action": "a", "desc": "2", "depends_on": ["s1"]},  # 合法
        ]}, goal="g", chat_id="c", user_id="u", valid_actions={"a"})
        assert p.steps[0].depends_on == []
        assert p.steps[1].depends_on == ["s1"]

    def test_verify_whitelist(self):
        p = parse_plan({"steps": [
            {"id": "s1", "action": "a", "desc": "1", "verify": "乱写"},
            {"id": "s2", "action": "a", "desc": "2", "verify": "llm_judge"},
        ]}, goal="g", chat_id="c", user_id="u", valid_actions={"a"})
        assert p.steps[0].verify == "tool_ok"
        assert p.steps[1].verify == "llm_judge"

    def test_max_steps_truncated(self):
        payload = {"steps": [{"id": f"s{i}", "action": "a", "desc": str(i)}
                             for i in range(10)]}
        p = parse_plan(payload, goal="g", chat_id="c", user_id="u",
                       valid_actions={"a"}, max_steps=6)
        assert len(p.steps) == 6


# ---------- 执行循环 ----------

class TestRunLoop:
    @pytest.mark.asyncio
    async def test_happy_path_tool_then_synth(self, harness, monkeypatch):
        calls = []
        _bind_tools(monkeypatch, [_StubTool("web_search",
                                            lambda a: calls.append(a) or "搜索结果")])
        plan = TaskPlan(goal="调研并汇总", chat_id="qq:g1:group", steps=[
            Step(id="s1", action="web_search", desc="搜资料", args_hint={"q": "AI"}),
            Step(id="s2", action=SYNTH_ACTION, desc="汇总成报告", depends_on=["s1"]),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert plan.state == "done"
        assert all(s.status == "done" for s in plan.steps)
        assert plan.steps[0].result == "搜索结果"
        assert harness["sent"], "完成后必须主动汇报"
        oc = _outcomes("qq:g1:group")
        assert oc and oc[-1]["kind"] == "task_kernel" and oc[-1]["status"] == "done"

    @pytest.mark.asyncio
    async def test_retry_once_then_success(self, harness, monkeypatch):
        attempts = {"n": 0}

        def flaky(args):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("网络抖动")
            return "成了"

        _bind_tools(monkeypatch, [_StubTool("web_search", flaky)])
        plan = TaskPlan(goal="g", chat_id="qq:g2:group",
                        steps=[Step(id="s1", action="web_search", desc="搜")])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert attempts["n"] == 2, "失败必须重试一次"
        assert plan.state == "done"

    @pytest.mark.asyncio
    async def test_replan_after_repeated_failure(self, harness, monkeypatch):
        def always_fail(args):
            raise RuntimeError("彻底不行")

        _bind_tools(monkeypatch, [_StubTool("bad_tool", always_fail)])
        import junjun_agent.task_kernel.planner as planner

        async def fake_revise(plan, desc, err):
            return [Step(id="r1", action=SYNTH_ACTION, desc="直接根据已有信息收尾")]

        monkeypatch.setattr(planner, "revise_remaining", fake_revise)
        plan = TaskPlan(goal="g", chat_id="qq:g3:group",
                        steps=[Step(id="s1", action="bad_tool", desc="会失败的步骤")])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert plan.replans == 1
        assert plan.state == "done", "重规划换 synth 收尾后应完成"
        assert [s.id for s in plan.steps] == ["r1"]

    @pytest.mark.asyncio
    async def test_abort_when_replan_exhausted(self, harness, monkeypatch):
        monkeypatch.setattr(executor, "_cfg",
                            lambda: {"enable": True, "max_replans": 0})

        def always_fail(args):
            raise RuntimeError("不行")

        _bind_tools(monkeypatch, [_StubTool("bad_tool", always_fail)])
        plan = TaskPlan(goal="g", chat_id="qq:g4:group",
                        steps=[Step(id="s1", action="bad_tool", desc="必败步骤")])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert plan.state == "failed"
        oc = _outcomes("qq:g4:group")
        assert oc[-1]["status"] == "failed"
        assert "必败步骤" in oc[-1]["detail"] or "失败" in oc[-1]["detail"]

    @pytest.mark.asyncio
    async def test_deadline_aborts(self, harness, monkeypatch):
        _bind_tools(monkeypatch, [])
        plan = TaskPlan(goal="g", chat_id="qq:g5:group",
                        steps=[Step(id="s1", action=SYNTH_ACTION, desc="合成")])
        plan.deadline_ts = 1.0  # 早已过期
        await executor.kernel._run(plan)
        assert plan.state == "failed" and "时限" in plan.note

    @pytest.mark.asyncio
    async def test_llm_judge_rejects_bad_output(self, harness, monkeypatch):
        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "很水的结果")])
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model",
                            lambda slot="utils": _FakeModel("不行，内容太空"))
        plan = TaskPlan(goal="g", chat_id="qq:g6:group", steps=[
            Step(id="s1", action="web_search", desc="深度调研", verify="llm_judge"),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        # 验收两次都不过 -> 重试后 replan（planner 真调会失败返回 None）-> failed
        assert plan.state == "failed"


# ---------- Langfuse 指标（方案 §六：plan 为 trace，步骤为 span） ----------

class _RecSpan:
    """记录型假 span（lf 未启用时生产用 NoopSpan，这里验证指标字段本身）。"""

    def __init__(self, sink, **kw):
        self.kw = kw
        self.updates = []
        self._sink = sink

    def __enter__(self):
        self._sink.append(self)
        return self

    def __exit__(self, *e):
        return False

    def update(self, **kw):
        self.updates.append(kw)


def _fake_lf(monkeypatch):
    spans = []
    import junjun_core.observability as obs
    monkeypatch.setattr(obs, "lf", SimpleNamespace(
        start_span=lambda **kw: _RecSpan(spans, **kw), enabled=True))
    return spans


class TestTracing:
    @pytest.mark.asyncio
    async def test_root_and_step_spans_recorded(self, harness, monkeypatch):
        spans = _fake_lf(monkeypatch)
        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "搜索结果")])
        plan = TaskPlan(goal="调研并汇总", chat_id="qq:t1:group", steps=[
            Step(id="s1", action="web_search", desc="搜资料"),
            Step(id="s2", action=SYNTH_ACTION, desc="汇总", depends_on=["s1"]),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        root = next(s for s in spans if s.kw["name"].startswith("task_kernel."))
        steps = [s for s in spans if s.kw["name"].startswith("task_kernel_step.")]
        assert root.kw["metadata"]["steps_total"] == 2
        assert len(steps) == 2, "每步一个 span"
        final = root.updates[-1]["metadata"]
        assert final["state"] == "done" and final["steps_done"] == 2
        assert final["steps_failed"] == 0 and final["verify_failures"] == 0
        assert "duration_s" in final

    @pytest.mark.asyncio
    async def test_verify_failures_counted(self, harness, monkeypatch):
        spans = _fake_lf(monkeypatch)
        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "很水的结果")])
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model",
                            lambda slot="utils": _FakeModel("不行，内容太空"))
        plan = TaskPlan(goal="g", chat_id="qq:t2:group", steps=[
            Step(id="s1", action="web_search", desc="深度调研", verify="llm_judge"),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        root = next(s for s in spans if s.kw["name"].startswith("task_kernel."))
        final = root.updates[-1]["metadata"]
        assert final["state"] == "failed"
        assert final["verify_failures"] == 2, "首次 + 重试各验收失败一次"

    @pytest.mark.asyncio
    async def test_noop_when_lf_disabled(self, harness, monkeypatch):
        """lf 空操作（NoopSpan）时执行循环行为完全不变——tracing 不能影响主流程。"""
        import junjun_core.observability as obs
        from junjun_core.observability.langfuse_client import _NoopSpan
        monkeypatch.setattr(obs, "lf", SimpleNamespace(
            start_span=lambda **kw: _NoopSpan(), enabled=False))
        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "结果")])
        plan = TaskPlan(goal="g", chat_id="qq:t3:group",
                        steps=[Step(id="s1", action="web_search", desc="搜")])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert plan.state == "done"


# ---------- 接单入口 ----------

class TestTrySubmit:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setattr(executor, "_cfg", lambda: {"enable": False})
        import junjun_agent.task_kernel.planner as planner

        async def boom(*a, **k):
            raise AssertionError("开关关闭时不许调规划器")

        monkeypatch.setattr(planner, "make_plan", boom)
        assert await executor.kernel.try_submit("帮我调研写报告", chat_id="qq:x:group") is None

    @pytest.mark.asyncio
    async def test_planner_failure_falls_back(self, harness, monkeypatch):
        import junjun_agent.task_kernel.planner as planner

        async def none_plan(*a, **k):
            return None

        monkeypatch.setattr(planner, "make_plan", none_plan)
        assert await executor.kernel.try_submit("帮我调研写报告", chat_id="qq:x:group") is None

    @pytest.mark.asyncio
    async def test_accept_returns_ack(self, harness, monkeypatch):
        import junjun_agent.task_kernel.planner as planner

        async def fake_plan(goal, **k):
            return TaskPlan(goal=goal, chat_id=k.get("chat_id", ""),
                            steps=[Step(id="s1", action=SYNTH_ACTION, desc="合成")])

        monkeypatch.setattr(planner, "make_plan", fake_plan)

        async def fake_run(self, plan):
            plan.state = "done"

        monkeypatch.setattr(executor.TaskKernel, "_run", fake_run)
        ack = await executor.kernel.try_submit("帮我调研写报告", chat_id="qq:g7:group")
        assert ack and isinstance(ack, str)
        await asyncio.sleep(0)  # 让 create_task 的收尾跑掉
