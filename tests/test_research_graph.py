"""deep_research LangGraph 引擎测试：正常流 / 反思轮 / 空材料 / 崩溃续跑 / 启动恢复补交付。

research._plan/_collect/_replan/_synthesize 全部打桩（节点直接调它们），
不触真实 LLM/搜索；sqlite 用 tmp_path 独立文件，不触生产库。
"""

import json

import pytest

import junjun_core.config.config as cfg_mod
from junjun_skills.plugins.async_task import research, research_graph

CHAT = "qq:12345:group"


def _set_config(raw: dict):
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw=raw)


def _job(job_id="job-abc", title="研究主题"):
    return type("J", (), {"job_id": job_id, "chat_id": CHAT,
                          "title": title, "kind": "deep_research"})()


class _Stubs:
    """research 纯函数的替身：计数 + 可切行为模式。"""

    def __init__(self):
        self.plan_calls = 0
        self.collect_calls = 0
        self.replan_calls = 0
        self.synth_calls = 0
        self.searched_queries: list = []
        self.collect_mode = "rich"   # rich | thin_then_rich | empty
        self.replan_result = ["新查询A"]

    async def plan(self, topic, model=None):
        self.plan_calls += 1
        return ["q1", "q2"]

    async def collect(self, queries, *, search=None, fetch=None):
        self.collect_calls += 1
        self.searched_queries.append(list(queries))
        if self.collect_mode == "empty":
            return []
        if self.collect_mode == "thin_then_rich" and self.collect_calls == 1:
            return [{"title": "薄", "url": "http://old", "snippet": "s",
                     "content": ""}]
        # URL 带轮次后缀，跨轮去重不会把第二轮误吞
        return [{"title": f"t-{q}", "url": f"http://{q}-{self.collect_calls}",
                 "snippet": "s", "content": "全文"} for q in queries]

    async def replan(self, topic, old, got, model=None):
        self.replan_calls += 1
        return list(self.replan_result)

    async def synth(self, topic, items, model=None):
        self.synth_calls += 1
        return "最终报告"


@pytest.fixture
def env(tmp_path, monkeypatch):
    old = cfg_mod.global_config
    _set_config({"deep_research": {"engine": "langgraph", "queries": 3,
                                   "pages_per_query": 2, "fetch_max_chars": 100,
                                   "report_max_chars": 500, "max_rounds": 2,
                                   "min_items": 2, "min_fulltext": 2}})
    stubs = _Stubs()
    monkeypatch.setattr(research, "_plan", stubs.plan)
    monkeypatch.setattr(research, "_collect", stubs.collect)
    monkeypatch.setattr(research, "_replan", stubs.replan)
    monkeypatch.setattr(research, "_synthesize", stubs.synth)
    research_graph._persist_dir = tmp_path
    research_graph._graph = None
    research_graph._recovered = False
    yield stubs
    research_graph._graph = None
    research_graph._persist_dir = None
    research_graph._recovered = False
    cfg_mod.global_config = old


async def _close_graph():
    """sqlite 连接随事件循环关闭会刷 warning，测完显式关。"""
    g = research_graph._graph
    cp = getattr(g, "checkpointer", None)
    conn = getattr(cp, "conn", None)
    if conn is not None:
        await conn.close()


def _registry_dict(tmp_path) -> dict:
    p = tmp_path / "active_research.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class TestRun:
    @pytest.mark.asyncio
    async def test_happy_path(self, env, tmp_path):
        from langgraph.checkpoint.memory import MemorySaver
        research_graph._graph = research_graph.build_graph(MemorySaver())
        out = await research_graph.run(_job(), "绝区零配队")
        assert out == "最终报告"
        assert env.plan_calls == 1 and env.collect_calls == 1
        assert env.replan_calls == 0, "材料充足不该触发反思"
        assert env.synth_calls == 1
        assert _registry_dict(tmp_path) == {}, "完成后注册表必须摘除"

    @pytest.mark.asyncio
    async def test_reflect_round(self, env):
        """首轮材料薄 -> 反思改写 -> 第二轮合并 -> 综述。"""
        from langgraph.checkpoint.memory import MemorySaver
        research_graph._graph = research_graph.build_graph(MemorySaver())
        env.collect_mode = "thin_then_rich"
        out = await research_graph.run(_job(), "宁德到深圳怎么去")
        assert out == "最终报告"
        assert env.collect_calls == 2 and env.replan_calls == 1
        assert env.searched_queries[1] == ["新查询A"]

    @pytest.mark.asyncio
    async def test_empty_materials_raises_and_cleans_registry(self, env, tmp_path):
        from langgraph.checkpoint.memory import MemorySaver
        research_graph._graph = research_graph.build_graph(MemorySaver())
        env.collect_mode = "empty"
        with pytest.raises(RuntimeError, match="没查到"):
            await research_graph.run(_job(), "查不到的东西")
        assert env.collect_calls == 2, "空材料也要走满反思轮再认输"
        assert _registry_dict(tmp_path) == {}, "失败也要摘注册表（结局由 runner 上报）"


class TestCrashResume:
    """sqlite 断点：图执行到一半被掐（recursion_limit 模拟崩溃），
    重启后同 thread_id 传 None 从断点续跑，已完成的节点不重跑。"""

    @pytest.mark.asyncio
    async def test_resume_skips_finished_nodes(self, env):
        try:
            graph = await research_graph._ensure_graph()
            cfg_low = {"configurable": {"thread_id": "job-crash"},
                       "recursion_limit": 2}
            from langgraph.errors import GraphRecursionError
            with pytest.raises(GraphRecursionError):
                await graph.ainvoke({"topic": "主题", "chat_id": CHAT,
                                     "job_id": "job-crash"}, cfg_low)
            assert env.plan_calls == 1
            # 续跑（等价于重启后 recover 的 ainvoke(None)）
            state = await graph.ainvoke(
                None, research_graph._thread_cfg("job-crash"))
            assert state["report"] == "最终报告"
            assert env.plan_calls == 1, "规划已落 checkpoint，续跑不得重规划"
            assert env.collect_calls == 1, "首轮检索已落 checkpoint，续跑不得重搜"
        finally:
            await _close_graph()


class TestRecover:
    """启动恢复：注册表里的任务断点续跑 + 补交付（结局登记 + 口吻播报）。"""

    def _patch_delivery(self, monkeypatch):
        delivered = {"outcomes": [], "voiced": []}
        from junjun_agent.tasks import task_manager
        monkeypatch.setattr(task_manager, "_record_outcome",
                            lambda *a, **kw: delivered["outcomes"].append(a))
        async def fake_voice(*a, **kw):
            delivered["voiced"].append(a)
            return "已播报"
        monkeypatch.setattr(task_manager, "_voice_outcome", fake_voice)
        return delivered

    @pytest.mark.asyncio
    async def test_recover_resumes_and_delivers(self, env, monkeypatch, tmp_path):
        """进程在 collect 后崩了：重启 recover -> 续跑完 -> 补交付。"""
        delivered = self._patch_delivery(monkeypatch)
        try:
            graph = await research_graph._ensure_graph()
            from langgraph.errors import GraphRecursionError
            with pytest.raises(GraphRecursionError):
                await graph.ainvoke({"topic": "主题X", "chat_id": CHAT,
                                     "job_id": "job-r"},
                                    {"configurable": {"thread_id": "job-r"},
                                     "recursion_limit": 2})
            research_graph.registry_add("job-r", CHAT, "主题X")
            # 模拟重启：关掉图和连接，重新从 sqlite 建
            await _close_graph()
            research_graph._graph = None
            research_graph._recovered = False

            await research_graph.recover()
            assert _registry_dict(tmp_path) == {}
            assert delivered["voiced"], "续跑完成必须补播报"
            assert delivered["voiced"][0][3] == "最终报告"
            assert delivered["outcomes"], "结局必须登记（防模型失忆）"
            assert env.plan_calls == 1 and env.collect_calls == 1
        finally:
            await _close_graph()

    @pytest.mark.asyncio
    async def test_recover_completed_but_lingering_registry(self, env,
                                                            monkeypatch,
                                                            tmp_path):
        """图已跑完但注册表没来得及摘（崩溃窗口）：读快照补交付，不重复执行。"""
        delivered = self._patch_delivery(monkeypatch)
        try:
            graph = await research_graph._ensure_graph()
            state = await graph.ainvoke(
                {"topic": "主题Y", "chat_id": CHAT, "job_id": "job-done"},
                research_graph._thread_cfg("job-done"))
            assert state["report"] == "最终报告"
            research_graph.registry_add("job-done", CHAT, "主题Y")
            await _close_graph()
            research_graph._graph = None
            research_graph._recovered = False

            await research_graph.recover()
            assert _registry_dict(tmp_path) == {}
            assert len(delivered["voiced"]) == 1, "已完成的只补交付一次"
            assert env.synth_calls == 1, "恢复不得重跑综述"
        finally:
            await _close_graph()

    @pytest.mark.asyncio
    async def test_recover_idempotent(self, env):
        """recover 只跑一次（run_junjun 与手动调用并发时安全）。"""
        await research_graph.recover()
        research_graph._registry_save({})
        await research_graph.recover()  # 第二次直接返回


class TestHandlerRouting:
    """deep_research_handler 的引擎分流：langgraph 且无测试桩 -> 图引擎。"""

    @pytest.mark.asyncio
    async def test_langgraph_engine_routes_to_graph(self, env, monkeypatch):
        calls = []

        async def fake_run(job, topic):
            calls.append((job.job_id, topic))
            return "图报告"
        monkeypatch.setattr(research_graph, "run", fake_run)
        out = await research.deep_research_handler(_job(), {"topic": "主题"})
        assert out == "图报告"
        assert calls == [("job-abc", "主题")]
        assert env.plan_calls == 0, "分流到图引擎就不走 legacy 流水线"

    @pytest.mark.asyncio
    async def test_di_stubs_stay_on_legacy(self, env, monkeypatch):
        """注入测试桩时即使 engine=langgraph 也走 legacy（测试友好）。"""
        async def fake_run(job, topic):
            raise AssertionError("不该分流到图引擎")
        monkeypatch.setattr(research_graph, "run", fake_run)
        out = await research.deep_research_handler(
            _job(), {"topic": "主题"},
            plan_model=env,  # 任意非 None 桩
            search=None, fetch=None)
        # legacy 路径会调被打桩的 research._plan（忽略传入的桩对象），最终出报告
        assert out == "最终报告"

    @pytest.mark.asyncio
    async def test_legacy_default_not_routed(self, tmp_path, monkeypatch):
        """没配 engine（默认 legacy）时不走图引擎。"""
        old = cfg_mod.global_config
        _set_config({"deep_research": {"max_rounds": 1, "min_items": 2,
                                       "min_fulltext": 2}})
        async def fake_run(job, topic):
            raise AssertionError("legacy 引擎不该走图")
        monkeypatch.setattr(research_graph, "run", fake_run)
        stubs = _Stubs()
        monkeypatch.setattr(research, "_plan", stubs.plan)
        monkeypatch.setattr(research, "_collect", stubs.collect)
        monkeypatch.setattr(research, "_synthesize", stubs.synth)
        try:
            out = await research.deep_research_handler(_job(), {"topic": "主题"})
            assert out == "最终报告"
            assert stubs.plan_calls == 1
        finally:
            cfg_mod.global_config = old
