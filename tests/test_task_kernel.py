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
    def __init__(self, name, fn, schema=None):
        self.name = name
        self.description = f"{name} 桩"
        self._fn = fn
        if schema is not None:
            self.args_schema = schema

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

    def test_done_criteria_carried(self):
        """Phase 1：完成判据从规划 JSON 落到 Step。"""
        payload = {"steps": [{"id": "s1", "action": "a", "desc": "d",
                              "done_criteria": "含三个要点"}]}
        p = parse_plan(payload, goal="g", chat_id="c", user_id="u",
                       valid_actions={"a"})
        assert p.steps[0].done_criteria == "含三个要点"

    def test_old_save_without_criteria_backcompat(self):
        """向后兼容：旧存档无 done_criteria 字段 -> 恢复为 ""，不炸。"""
        p = TaskPlan.from_dict({"goal": "g", "chat_id": "c",
                                "steps": [{"id": "s1", "action": "a", "desc": "d"}]})
        assert p.steps[0].done_criteria == ""


class TestParsePlanAsyncGuard:
    """异步接单工具的依赖步骤硬剔除（2026-08-14 生产实锤：deep_research→
    llm_synthesize 两步计划，合成步拿着接单回执凭空写报告先发群，82 秒后
    后台真报告又到，群里双份内容打架）。"""

    _ASYNC = frozenset({"deep_research"})

    def _parse(self, steps, **kw):
        return parse_plan({"steps": steps}, goal="g", chat_id="c", user_id="u",
                          async_actions=self._ASYNC, **kw)

    def test_async_dependent_dropped(self):
        p = self._parse([
            {"id": "s1", "action": "deep_research", "desc": "调研"},
            {"id": "s2", "action": SYNTH_ACTION, "desc": "写报告",
             "depends_on": ["s1"]},
        ], valid_actions={"deep_research"})
        assert [s.id for s in p.steps] == ["s1"]

    def test_async_ref_in_args_dropped(self):
        """不声明 depends_on、只在 args_hint 里 $引用 的一样剔。"""
        p = self._parse([
            {"id": "s1", "action": "deep_research", "desc": "调研"},
            {"id": "s2", "action": SYNTH_ACTION, "desc": "写",
             "args_hint": {"topic": "$s1 的材料"}},
        ], valid_actions={"deep_research"})
        assert [s.id for s in p.steps] == ["s1"]

    def test_async_transitive_dropped(self):
        p = self._parse([
            {"id": "s1", "action": "deep_research", "desc": "调研"},
            {"id": "s2", "action": SYNTH_ACTION, "desc": "写", "depends_on": ["s1"]},
            {"id": "s3", "action": "send_message", "desc": "转达",
             "depends_on": ["s2"]},
        ], valid_actions={"deep_research", "send_message"})
        assert [s.id for s in p.steps] == ["s1"]

    def test_independent_step_kept(self):
        """误判方向：与异步步骤无依赖的独立步骤不许误剔。"""
        p = self._parse([
            {"id": "s1", "action": "deep_research", "desc": "调研"},
            {"id": "s2", "action": "set_reminder", "desc": "顺便设个提醒"},
        ], valid_actions={"deep_research", "set_reminder"})
        assert [s.id for s in p.steps] == ["s1", "s2"]

    def test_sync_chain_untouched(self):
        """误判方向：普通同步工具链（搜索→汇总）不受异步守卫影响。"""
        p = self._parse([
            {"id": "s1", "action": "web_search", "desc": "搜"},
            {"id": "s2", "action": SYNTH_ACTION, "desc": "汇总",
             "depends_on": ["s1"], "args_hint": {"q": "$s1"}},
        ], valid_actions={"web_search"})
        assert [s.id for s in p.steps] == ["s1", "s2"]

    def test_no_async_actions_backcompat(self):
        """不传 async_actions 维持旧行为（依赖保留）。"""
        p = parse_plan({"steps": [
            {"id": "s1", "action": "deep_research", "desc": "调研"},
            {"id": "s2", "action": SYNTH_ACTION, "desc": "写",
             "depends_on": ["s1"]},
        ]}, goal="g", chat_id="c", user_id="u", valid_actions={"deep_research"})
        assert [s.id for s in p.steps] == ["s1", "s2"]


class TestAsyncJobTagContract:
    """tags 字面量契约：插件侧打标与内核侧识别必须同字（跨包没法共享常量，
    用测试钉死）。"""

    def test_plugin_tools_carry_tag(self):
        from junjun_agent.task_kernel.plan import ASYNC_JOB_TAG
        from junjun_skills.plugins.async_task import tools as at
        from junjun_skills.plugins.bilibili import tools as bt
        for t in (at.deep_research, at.run_background_task, bt.watch_video):
            assert ASYNC_JOB_TAG in (getattr(t, "tags", None) or []), t.name

    def test_catalog_marks_async(self, monkeypatch):
        from junjun_agent.task_kernel import planner
        import junjun_skills.registry as reg
        sync_tool = _StubTool("web_search", lambda a: "x")
        async_tool = _StubTool("deep_research", lambda a: "x")
        async_tool.tags = ["async_job"]
        monkeypatch.setattr(reg, "get_tools", lambda: [sync_tool, async_tool])
        cat = planner._tool_catalog()
        async_line = [l for l in cat.splitlines() if "deep_research" in l][0]
        sync_line = [l for l in cat.splitlines() if "web_search" in l][0]
        assert "［异步接单" in async_line
        assert "［异步接单" not in sync_line


class TestReportRemembered:
    @pytest.mark.asyncio
    async def test_final_report_remembered(self, harness, monkeypatch):
        """终态汇报必须 remember=True（2026-08-14 实锤：发到群里的调研报告
        既没落库也没进短期记忆，bot 被追问「报告呢」只能装傻——P1-6 同类）。
        async_jobs 汇报路径本就 remember=True，内核这条是漏开的孤岛。"""
        import junjun_agent.outbound as outbound
        cap = {}

        async def _cap(chat_id, segments, *, source="", remember=True):
            cap["remember"] = remember
            return True

        monkeypatch.setattr(outbound, "send_proactive", _cap)
        plan = TaskPlan(goal="调研", chat_id="qq:g1:group", state="done",
                        steps=[Step(id="s1", action="a", desc="d",
                                    status="done", result="成果")])
        await executor.kernel._report(plan)
        assert cap.get("remember") is True


# ---------- 执行循环 ----------

class TestUserIdSafetyNet:
    """身份参数兜底（2026-08-15 eval chain-weather-remind 实锤）：规划器看不到
    下单上下文，user_id 必填参数只能瞎编或丢整步。非数字占位值换成发起者；
    数字 QQ 保留（戳别人/改别人资料是合法定向用法，不许误伤）。"""

    @staticmethod
    def _schema():
        from pydantic import BaseModel

        class _S(BaseModel):
            content: str
            user_id: str

        return _S

    @staticmethod
    def _mk(plan_user_id, arg_user_id, **kw):
        got = {}
        schema = TestUserIdSafetyNet._schema()
        tool = _StubTool("set_reminder",
                         lambda a: got.update(a) or "ok", schema=schema)
        plan = TaskPlan(goal="提醒", chat_id="qq:u1:private",
                        user_id=plan_user_id,
                        steps=[Step(id="s1", action="set_reminder", desc="设提醒",
                                    args_hint={"content": "带伞",
                                               "user_id": arg_user_id})])
        return plan, got, tool

    @pytest.mark.asyncio
    async def test_placeholder_replaced(self, harness, monkeypatch):
        plan, got, tool = self._mk("3155572670", "auto")
        _bind_tools(monkeypatch, [tool])
        await executor.kernel._call_tool(plan, plan.steps[0])
        assert got["user_id"] == "3155572670"

    @pytest.mark.asyncio
    async def test_empty_filled(self, harness, monkeypatch):
        plan, got, tool = self._mk("3155572670", "")
        _bind_tools(monkeypatch, [tool])
        await executor.kernel._call_tool(plan, plan.steps[0])
        assert got["user_id"] == "3155572670"

    @pytest.mark.asyncio
    async def test_real_qq_kept(self, harness, monkeypatch):
        """数字 QQ 是合法定向（帮我戳张三）——不得被发起者覆盖（误判回归）。"""
        plan, got, tool = self._mk("3155572670", "2485424686")
        _bind_tools(monkeypatch, [tool])
        await executor.kernel._call_tool(plan, plan.steps[0])
        assert got["user_id"] == "2485424686"

    @pytest.mark.asyncio
    async def test_no_plan_user_untouched(self, harness, monkeypatch):
        plan, got, tool = self._mk("", "auto")
        _bind_tools(monkeypatch, [tool])
        await executor.kernel._call_tool(plan, plan.steps[0])
        assert got["user_id"] == "auto"   # 发起者未知：不编不盖，原样放行

    @pytest.mark.asyncio
    async def test_tool_without_user_id_untouched(self, harness, monkeypatch):
        from pydantic import BaseModel

        class _S(BaseModel):
            q: str

        got = {}
        tool = _StubTool("web_search", lambda a: got.update(a) or "ok",
                         schema=_S)
        plan = TaskPlan(goal="g", chat_id="qq:u2:private", user_id="3155572670",
                        steps=[Step(id="s1", action="web_search", desc="搜",
                                    args_hint={"q": "x"})])
        _bind_tools(monkeypatch, [tool])
        await executor.kernel._call_tool(plan, plan.steps[0])
        assert got == {"q": "x"}   # 无 user_id 字段的工具不许被塞参数


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

    @pytest.mark.asyncio
    async def test_registry_error_text_counts_as_failure(self, harness, monkeypatch):
        """注册表包装的错误文本必须判失败，不能当产出放行。

        registry._wrap_error_feedback 的现役格式是 "[TOOL_ERROR kind=...]"；
        executor 曾只认 "[TOOL_ERROR]" 精确前缀——错误文本当结果、步骤假成功、
        下游 llm_synthesize 拿错误串当前序素材（2026-08-12 golden_tasks 实锤）。
        """
        import junjun_agent.task_kernel.planner as planner

        async def fake_revise(plan, desc, err):
            return None  # 重规划也救不回，直接认输（不调 LLM）

        monkeypatch.setattr(planner, "revise_remaining", fake_revise)
        _bind_tools(monkeypatch, [_StubTool(
            "web_search",
            lambda a: "[TOOL_ERROR kind=timeout] 工具 web_search 网络超时")])
        plan = TaskPlan(goal="g", chat_id="qq:g5:group",
                        steps=[Step(id="s1", action="web_search", desc="搜")])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert plan.steps[0].status == "failed", "错误文本不得判为步骤成功"
        assert "TOOL_ERROR" in plan.steps[0].error
        assert plan.state == "failed"

    @pytest.mark.asyncio
    async def test_side_effect_serial_after_safe(self, harness, monkeypatch):
        """Phase 1 硬校验：无副作用步骤并行，副作用/成品步骤串行殿后。"""
        order = []
        _bind_tools(monkeypatch, [
            _StubTool("ai_draw", lambda a: order.append("draw") or "图已发"),
            _StubTool("web_search", lambda a: order.append("search") or "结果"),
        ])
        plan = TaskPlan(goal="g", chat_id="qq:g6:group", steps=[
            Step(id="s1", action="ai_draw", desc="画图"),   # 故意把成品写前面
            Step(id="s2", action="web_search", desc="搜"),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert order == ["search", "draw"], "副作用步骤必须排在安全步骤之后"
        assert plan.state == "done"

    @pytest.mark.asyncio
    async def test_judge_prompt_carries_done_criteria(self, harness, monkeypatch):
        """llm_judge 验收提示必须带完成判据（对准规划意图）。"""
        seen = []

        class _CapModel:
            async def ainvoke(self, msgs, config=None):
                seen.append(str(msgs[-1].content))
                return AIMessage(content="可以")

        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model",
                            lambda slot="utils": _CapModel())
        plan = TaskPlan(goal="g", chat_id="qq:g7:group", steps=[
            Step(id="s1", action=SYNTH_ACTION, desc="写报告", verify="llm_judge",
                 done_criteria="含三个要点"),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert any("完成判据" in p and "含三个要点" in p for p in seen), \
            "验收提示没带完成判据"

    @pytest.mark.asyncio
    async def test_max_replans_defaults_to_three(self, harness, monkeypatch):
        """Phase 1：max_replans 默认 1 -> 3（配置缺失时）。"""
        monkeypatch.setattr(executor, "_cfg", lambda: {
            "enable": True, "max_steps": 6, "deadline_minutes": 30,
            "replan_backoff_seconds": 0})
        _bind_tools(monkeypatch, [_StubTool(
            "bad_tool", lambda a: (_ for _ in ()).throw(RuntimeError("不行")))])
        import junjun_agent.task_kernel.planner as planner
        revise_calls = {"n": 0}

        async def fake_revise(plan, desc, err):
            revise_calls["n"] += 1
            return [Step(id=f"r{revise_calls['n']}", action="bad_tool",
                         desc="换个法子还是败")]

        monkeypatch.setattr(planner, "revise_remaining", fake_revise)
        plan = TaskPlan(goal="g", chat_id="qq:g8:group",
                        steps=[Step(id="s1", action="bad_tool", desc="必败步骤")])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert revise_calls["n"] == 3, "默认应重规划 3 次才认输"
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

    @pytest.mark.asyncio
    async def test_judge_exception_failopen_leaves_trace(self, harness, monkeypatch,
                                                         capsys):
        """2026-08-13 审查 P1：验收调用异常按通过处理（fail-open 语义保留，
        工具产出已在不当失败），但必须留痕——plan.verify_skipped 计数 +
        warning + 终态汇报点名，否则 key 故障时全步骤「验收通过」汇报成功。
        """
        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "搜索结果")])

        class _BoomJudge:
            async def ainvoke(self, msgs, config=None):
                if "只答「可以」" in str(msgs[-1].content):
                    raise RuntimeError("key 炸了")
                return AIMessage(content="汇报文本")

        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model",
                            lambda slot="utils": _BoomJudge())
        plan = TaskPlan(goal="g", chat_id="qq:g9:group", steps=[
            Step(id="s1", action="web_search", desc="深度调研", verify="llm_judge"),
        ])
        plan.deadline_ts = 9e18
        await executor.kernel._run(plan)
        assert plan.state == "done", "验收异常应 fail-open，步骤产出已在不当失败"
        assert plan.verify_skipped == 1
        sent = "\n".join(str(seg.data) for _, segs, _ in harness["sent"]
                         for seg in segs)
        assert "验收环节没跑通" in sent, "终态汇报必须点名未经验收的步骤"
        assert "验收调用异常" in capsys.readouterr().out
        # 持久化往返不丢；旧存档无此字段 -> 0 不炸
        assert TaskPlan.from_dict(plan.to_dict()).verify_skipped == 1
        old = TaskPlan.from_dict({"goal": "g", "chat_id": "c", "steps": []})
        assert old.verify_skipped == 0


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
    async def test_planner_failure_ack_then_honest_followup(self, harness, monkeypatch):
        """新契约（2026-08-13 接单前置）：规划失败不再返回 None 回退对话通道——
        接单话术已先回，后台规划落空必须补一句诚实交代（不许装死）。"""
        import junjun_agent.task_kernel.planner as planner

        async def none_plan(*a, **k):
            return None

        monkeypatch.setattr(planner, "make_plan", none_plan)
        ack = await executor.kernel.try_submit("帮我调研写报告", chat_id="qq:x:group")
        assert ack  # 先应声
        sent = harness["sent"]
        for _ in range(50):  # 等后台规划任务跑完（桩即时返回，只是过几个事件循环）
            if sent:
                break
            await asyncio.sleep(0.02)
        assert sent and sent[0][0] == "qq:x:group"
        assert "拆不动" in sent[0][1][0].data
        assert not executor.kernel._plans  # 拒收不留计划

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


# ---------- 规划器（Phase 1：schema 摘要 / 判据落步骤 / 垃圾输出重试一次） ----------

class TestPlannerBits:
    def test_catalog_has_schema_digest(self, monkeypatch):
        """工具清单必须带参数签名摘要——基线 3/10 失败死于规划器瞎猜参数名。"""
        from pydantic import BaseModel
        from junjun_agent.task_kernel import planner

        class _BiliArgs(BaseModel):
            url: str
            hint: str = ""

        tool = _StubTool("bilibili_summary", lambda a: "")
        tool.args_schema = _BiliArgs
        _bind_tools(monkeypatch, [tool])
        catalog = planner._tool_catalog()
        assert "bilibili_summary(url:str" in catalog
        assert "hint:str?" in catalog, "可选参数要带 ? 标记"

    @pytest.mark.asyncio
    async def test_make_plan_retries_once_on_garbage(self, monkeypatch):
        """首次产出非合法计划 -> 追加提醒重试一次（基线 2/10 失败死于单次 None）。"""
        from junjun_agent.task_kernel import planner
        outs = ["我觉得应该先搜一下再汇总……",
                '{"steps": [{"id": "s1", "action": "web_search", "desc": "搜",'
                ' "done_criteria": "拿到结果"}]}']

        class _SeqModel:
            async def ainvoke(self, msgs, config=None):
                return AIMessage(content=outs.pop(0))

        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "")])
        plan = await planner.make_plan("调研", chat_id="c", user_id="u",
                                       model=_SeqModel())
        assert plan is not None, "第二次产出合法必须接单"
        assert plan.steps[0].action == "web_search"
        assert plan.steps[0].done_criteria == "拿到结果"

    @pytest.mark.asyncio
    async def test_make_plan_gives_up_after_two(self, monkeypatch):
        from junjun_agent.task_kernel import planner
        outs = ["不会", "还是不会"]

        class _SeqModel:
            async def ainvoke(self, msgs, config=None):
                return AIMessage(content=outs.pop(0))

        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "")])
        plan = await planner.make_plan("调研", chat_id="c", user_id="u",
                                       model=_SeqModel())
        assert plan is None, "两次都废必须回退对话通道"

    def test_extract_json_bare_array_fallback(self):
        """裸数组产出兜底成 {"steps": [...]}——revise 提示事故期的 GLM 自由发挥形态。"""
        from junjun_agent.task_kernel import planner
        payload = planner._extract_json('```json\n[{"id": "r1", "task": "x"}]\n```')
        assert payload == {"steps": [{"id": "r1", "task": "x"}]}

    def test_reviser_prompt_shows_full_format(self):
        """重规划提示必须自带输出格式——「格式同前」是空引用（2026-08-12 实锤：
        GLM 没看过规划器提示，自由发挥成裸数组+自造字段名，revise 全军覆没）。"""
        from junjun_agent.task_kernel import planner
        assert '"steps"' in planner._REVISER_PROMPT
        assert '"action"' in planner._REVISER_PROMPT
        assert 'depends_on' in planner._REVISER_PROMPT

    def test_merge_revisal_keeps_unlisted_pending(self):
        """重规划未列出的原 pending 步骤必须保留——不声明就丢 = 目标静默放弃
        （2026-08-12 实锤：send_feed 人审步骤被吞，任务「完成」但说说没发）。"""
        from junjun_agent.task_kernel.plan import merge_revisal
        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="a", desc="d1", status="done", result="r"),
            Step(id="s2", action="b", desc="d2", status="failed"),
            Step(id="s3", action="send_feed", desc="发布", status="pending",
                 depends_on=["s2"], verify="human"),
        ])
        merge_revisal(plan, [Step(id="r1", action=SYNTH_ACTION, desc="收尾")])
        ids = [s.id for s in plan.steps]
        assert ids == ["s1", "r1", "s3"], "原 pending 不许悄悄丢"
        assert plan.steps[-1].verify == "human", "人审门必须保留"
        assert plan.steps[-1].depends_on == [], "断掉的依赖应剔除"

    def test_merge_revisal_honors_explicit_drop(self):
        """显式 drop 声明的步骤才允许放弃。"""
        from junjun_agent.task_kernel.planner import Revisal
        from junjun_agent.task_kernel.plan import merge_revisal
        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="a", desc="d1", status="done", result="r"),
            Step(id="s2", action="b", desc="d2", status="failed"),
            Step(id="s3", action="c", desc="d3", status="pending"),
        ])
        merge_revisal(plan, Revisal(
            [Step(id="r1", action=SYNTH_ACTION, desc="收尾")], drop=["s3"]))
        assert [s.id for s in plan.steps] == ["s1", "r1"]

    @pytest.mark.asyncio
    async def test_revise_parses_drop_list(self, monkeypatch):
        from junjun_agent.task_kernel import planner

        class _M:
            async def ainvoke(self, msgs, config=None):
                return AIMessage(content=(
                    '{"steps": [{"id": "r1", "action": "web_search", "desc": "重搜"}],'
                    ' "drop": ["s3"]}'))

        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "")])
        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="web_search", desc="d", status="done", result="r")])
        rev = await planner.revise_remaining(plan, "败了", "err", model=_M())
        assert rev is not None and rev.drop == ["s3"]
        assert [s.id for s in rev.steps] == ["r1"]

    @pytest.mark.asyncio
    async def test_revise_synth_material_backstop(self, monkeypatch):
        """重规划产出的 llm_synthesize 依赖被剥光时自动挂接已完成产出
        （2026-08-15 eval research-video-notes 实锤：reviser 把合成步骤依赖
        指向失败步骤 s2，剥离后零材料合成，摆烂输出「我没收到材料」）。"""
        from junjun_agent.task_kernel import planner

        class _M:
            async def ainvoke(self, msgs, config=None):
                return AIMessage(content=(
                    '{"steps": [{"id": "r1", "action": "llm_synthesize",'
                    ' "desc": "整理笔记", "depends_on": ["s2"]}]}'))  # s2 是失败步骤

        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "")])
        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="web_search", desc="搜", status="done", result="素材"),
            Step(id="s2", action="web_search", desc="败", status="failed"),
        ])
        rev = await planner.revise_remaining(plan, "败了", "err", model=_M())
        assert rev.steps[0].depends_on == ["s1"], "合成步骤断供必须自动挂已完成产出"

    @pytest.mark.asyncio
    async def test_revise_tool_step_no_backstop(self, monkeypatch):
        """误判回归：非合成步骤无依赖是正常形态，不许被挂依赖。"""
        from junjun_agent.task_kernel import planner

        class _M:
            async def ainvoke(self, msgs, config=None):
                return AIMessage(content=(
                    '{"steps": [{"id": "r1", "action": "web_search", "desc": "重搜",'
                    ' "depends_on": ["s9"]}]}'))   # s9 不存在 -> 剥光

        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "")])
        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="web_search", desc="搜", status="done", result="素材")])
        rev = await planner.revise_remaining(plan, "败了", "err", model=_M())
        assert rev.steps[0].depends_on == [], "工具步骤不许被自动挂依赖"

    @pytest.mark.asyncio
    async def test_revise_synth_valid_dep_kept(self, monkeypatch):
        """误判回归：合成步骤自己指对了已完成步骤，原样保留不被改写。"""
        from junjun_agent.task_kernel import planner

        class _M:
            async def ainvoke(self, msgs, config=None):
                return AIMessage(content=(
                    '{"steps": [{"id": "r1", "action": "llm_synthesize",'
                    ' "desc": "整理", "depends_on": ["s1"]}]}'))

        _bind_tools(monkeypatch, [_StubTool("web_search", lambda a: "")])
        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="web_search", desc="搜", status="done", result="素材"),
            Step(id="s2", action="web_search", desc="搜2", status="done", result="素材2")])
        rev = await planner.revise_remaining(plan, "败了", "err", model=_M())
        assert rev.steps[0].depends_on == ["s1"]

    def test_planner_prompt_scopes_llm_judge(self):
        """规划提示必须把 llm_judge 限在文本产出步骤——工具步骤配 llm_judge
        是逼工具返回它生产不了的东西（2026-08-12 watch_video 观后感判据必死实锤）。"""
        from junjun_agent.task_kernel import planner
        assert "给工具步骤配 llm_judge" in planner._PLANNER_PROMPT


class TestStepTimeout:
    """步骤级超时（Phase 2）：挂死的 LLM/工具调用按失败处理，retry/replan 接管。"""

    @pytest.mark.asyncio
    async def test_hung_step_fails_fast(self, monkeypatch):
        from junjun_agent.task_kernel import executor
        monkeypatch.setattr(executor, "_cfg", lambda: {"step_timeout_seconds": 0.05})
        plan = TaskPlan(goal="g", chat_id="c",
                        steps=[Step(id="s1", action="web_search", desc="x")])

        async def _hang(self, plan, step):
            import asyncio
            await asyncio.sleep(10)

        monkeypatch.setattr(executor.TaskKernel, "_run_step_inner", _hang)
        await executor.kernel._run_step(plan, plan.steps[0])
        assert plan.steps[0].status == "failed"
        assert "超时" in plan.steps[0].error
        assert executor.kernel_step_approved() is False  # 放行位没被超时路径泄漏


class TestStepIdentity:
    """步骤按 plan 发起者身份执行（2026-08-13 审查 P1 双案）：
    审批恢复复制管理员 context——步骤不许借管理员身份（跨会话隐私外泄）；
    断点恢复是空 context——workspace 工具不许落共享 unknown/。"""

    async def _run_and_capture(self, monkeypatch, plan):
        from junjun_core import security
        from junjun_skills.builtin.memory_skills import current_chat_id
        seen = {}

        async def _fake_call(self, plan, step):
            seen["user_id"] = security.current_user_id.get()
            seen["chat_id"] = current_chat_id.get()
            seen["privileged"] = security.is_admin_privileged()
            return "ok"

        async def _ok_verify(self, plan, step, result):
            return True

        monkeypatch.setattr(executor.TaskKernel, "_call_tool", _fake_call)
        monkeypatch.setattr(executor.TaskKernel, "_verify", _ok_verify)
        await executor.kernel._run_step(plan, plan.steps[0])
        assert plan.steps[0].status == "done"
        return seen

    @pytest.mark.asyncio
    async def test_non_admin_plan_in_admin_context(self, monkeypatch):
        """P1-1 案发现场：管理员批完一步，后续步骤必须回到提交者（非管理员）身份。"""
        from junjun_core import security
        from junjun_skills.builtin.memory_skills import current_chat_id
        monkeypatch.setenv("ADMIN_QQ", "99999")
        security.set_caller("99999", at_bot=True, is_group=False)  # 管理员审批 context
        tok = current_chat_id.set("qq:99999:private")
        try:
            plan = TaskPlan(goal="g", chat_id="qq:777:group", user_id="12345",
                            steps=[Step(id="s1", action="x", desc="d")])
            seen = await self._run_and_capture(monkeypatch, plan)
            assert seen["user_id"] == "12345"           # 不是管理员
            assert seen["chat_id"] == "qq:777:group"    # 不是管理员私聊
            assert seen["privileged"] is False          # 不继承审批特权
        finally:
            current_chat_id.reset(tok)
            security.set_caller("", at_bot=False, is_group=False)

    @pytest.mark.asyncio
    async def test_admin_group_plan_keeps_privilege(self, monkeypatch):
        """不误伤：管理员本人在群里派单（提交必经 @bot=显式指令）特权仍在。"""
        from junjun_core import security
        monkeypatch.setenv("ADMIN_QQ", "99999")
        security.set_caller("", at_bot=False, is_group=False)  # 模拟断点恢复空 context
        try:
            plan = TaskPlan(goal="g", chat_id="qq:777:group", user_id="99999",
                            steps=[Step(id="s1", action="x", desc="d")])
            seen = await self._run_and_capture(monkeypatch, plan)
            assert seen["user_id"] == "99999"
            assert seen["privileged"] is True
        finally:
            security.set_caller("", at_bot=False, is_group=False)


class TestApprovalTransparency:
    """审批必须带实际入参（2026-08-13 审查 P1）：只给 desc 是盲批。"""

    def test_fmt_step_args_truncates(self):
        from junjun_agent.task_kernel.graph import _fmt_step_args
        assert _fmt_step_args(None) == ""
        assert _fmt_step_args(Step(id="s", action="a", desc="d")) == ""
        s = Step(id="s", action="run_code", desc="d",
                 args_hint={"code": "print(1)"})
        assert _fmt_step_args(s) == '{"code": "print(1)"}'
        big = Step(id="s", action="run_code", desc="d",
                   args_hint={"code": "x" * 1000})
        out = _fmt_step_args(big)
        assert len(out) < 510 and out.endswith("……")

    @pytest.mark.asyncio
    async def test_notify_admin_includes_args(self, monkeypatch):
        from junjun_core import security
        from junjun_agent.task_kernel import graph as tk_graph
        sent = []

        async def _fake_notify(text):
            sent.append(text)
            return True

        monkeypatch.setattr(security, "notify_admin", _fake_notify)
        plan = TaskPlan(goal="统计表格", chat_id="qq:777:group")
        await tk_graph.runner._notify_admin(plan, {
            "goal": "统计表格",
            "step": {"id": "s2", "action": "run_code", "desc": "跑统计",
                     "args": '{"code": "print(1)"}'},
        })
        assert sent and "参数：" in sent[0] and "print(1)" in sent[0]

    @pytest.mark.asyncio
    async def test_notify_admin_no_args_no_line(self, monkeypatch):
        from junjun_core import security
        from junjun_agent.task_kernel import graph as tk_graph
        sent = []

        async def _fake_notify(text):
            sent.append(text)
            return True

        monkeypatch.setattr(security, "notify_admin", _fake_notify)
        plan = TaskPlan(goal="g", chat_id="c")
        await tk_graph.runner._notify_admin(plan, {
            "goal": "g", "step": {"id": "s1", "action": "web_search", "desc": "d"}})
        assert sent and "参数：" not in sent[0]
