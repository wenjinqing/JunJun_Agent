"""步骤材料库测试（2026-08-15，PTC「中间数据不进城」的低成本移植）。

核心断言：大产出全文落盘可回读、上下文只留摘要+指针、短产出行为不变、
落盘失败降级旧截断、合成步骤拿得到全文且总量有预算。
"""

import pytest
from langchain_core.messages import AIMessage

import junjun_agent.task_kernel.materials as materials
from junjun_agent.task_kernel import executor
from junjun_agent.task_kernel.plan import Step, TaskPlan


@pytest.fixture
def _tmp_materials(monkeypatch, tmp_path):
    """材料库指向临时目录——绝不碰 data/。"""
    monkeypatch.setattr(materials, "_dir", lambda: tmp_path)
    return tmp_path


class TestStoreResult:
    def test_short_result_stays_inline(self, _tmp_materials):
        stub, mid = materials.store_result("p1", "s1", "短结果")
        assert stub == "短结果" and mid == ""
        assert list(_tmp_materials.iterdir()) == []          # 没落盘

    def test_long_result_externalized(self, _tmp_materials):
        text = "甲" * 800
        stub, mid = materials.store_result("p1", "s1", text)
        assert mid == "p1/s1"
        assert len(stub) < 400                                # 上下文只留短摘要
        assert "全文 800 字已存材料" in stub                  # 指针可见
        assert materials.read(mid, 10000) == text             # 全文可回读

    def test_store_failure_degrades_to_inline(self, monkeypatch):
        monkeypatch.setattr(materials, "store", lambda *a: "")
        stub, mid = materials.store_result("p1", "s1", "乙" * 800)
        assert mid == "" and len(stub) == 500                 # 旧截断行为兜底

    def test_read_missing_returns_empty(self, _tmp_materials):
        assert materials.read("no/such", 100) == ""
        assert materials.read("", 100) == ""

    def test_read_caps_chars(self, _tmp_materials):
        mid = materials.store("p1", "s1", "丙" * 500)
        out = materials.read(mid, 100)
        assert out.startswith("丙" * 100) and "材料截断" in out


class TestExecutorIntegration:
    @pytest.mark.asyncio
    async def test_done_step_externalizes_and_synth_reads_full(
            self, _tmp_materials, monkeypatch):
        """长跑工具产出 → step.result 是指针摘要；llm_synthesize 提示里带全文。"""
        long_text = "数据明细" + "行" * 900

        class _Tool:
            name = "fetch_page"
            description = "桩"
            args_schema = None

            async def ainvoke(self, args):
                return long_text

        import junjun_skills.registry as reg
        monkeypatch.setattr(reg, "get_tools", lambda session=None: [_Tool()])

        captured = {}

        class _Model:
            async def ainvoke(self, msgs, config=None):
                captured["prompt"] = msgs[-1].content
                return AIMessage(content="报告")

        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot="utils": _Model())

        plan = TaskPlan(goal="g", chat_id="c", user_id="u", steps=[
            Step(id="s1", action="fetch_page", desc="抓全文"),
            Step(id="s2", action="llm_synthesize", desc="写报告",
                 depends_on=["s1"], verify="none"),
        ])
        k = executor.TaskKernel()
        await k._run_step(plan, plan.steps[0])
        s1 = plan.steps[0]
        assert s1.status == "done"
        assert s1.material_id == f"{plan.plan_id}/s1"
        assert len(s1.result) < 400 and "已存材料" in s1.result

        await k._run_step(plan, plan.steps[1])
        assert plan.steps[1].status == "done"
        assert long_text in captured["prompt"]                # 合成拿到全文

    @pytest.mark.asyncio
    async def test_synth_total_budget(self, _tmp_materials, monkeypatch):
        """材料总预算：三条 4000 字材料，总量 12000 封顶不烧穿。"""
        import junjun_llm
        captured = {}

        class _Model:
            async def ainvoke(self, msgs, config=None):
                captured["prompt"] = msgs[-1].content
                return AIMessage(content="x")

        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot="utils": _Model())
        plan = TaskPlan(goal="g", chat_id="c", user_id="u", steps=[
            Step(id=f"s{i}", action="a", desc=f"步骤{i}", status="done",
                 result=f"摘要{i}") for i in range(3)
        ])
        for s in plan.steps:  # 手动外置三条大材料
            s.material_id = materials.store(plan.plan_id, s.id, "料" * 5000)
        plan.steps.append(Step(id="sx", action="llm_synthesize", desc="合成",
                               depends_on=["s0", "s1", "s2"], verify="none"))
        await executor.TaskKernel()._synthesize(plan, plan.steps[-1])
        assert len(captured["prompt"]) < 13000

    def test_persistence_roundtrip_keeps_material_id(self):
        """落盘/恢复：material_id 字段随存档往返；旧存档无此字段兼容。"""
        plan = TaskPlan(goal="g", chat_id="c", user_id="u", steps=[
            Step(id="s1", action="a", desc="d", status="done",
                 result="摘要", material_id="p/s1")])
        back = TaskPlan.from_dict(plan.to_dict())
        assert back.steps[0].material_id == "p/s1"
        old = plan.to_dict()
        for s in old["steps"]:
            del s["material_id"]                              # 模拟旧存档
        back2 = TaskPlan.from_dict(old)
        assert back2.steps[0].material_id == ""
