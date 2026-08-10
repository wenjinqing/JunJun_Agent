"""deep_research 的 LangGraph 执行引擎：崩溃断点续跑（宁德事故 + 进程崩溃双驱动）。

背景：AsyncJob 任务体是内存态——进程重启 = 任务丢（task_manager 只把结局
标「进程重启，任务中断」，不重跑）。一次深度研究要烧几分钟搜索+LLM，
从头再来既浪费又可能再崩。本图把 plan→collect→reflect→synthesize 搬进
StateGraph + SqliteSaver，崩溃后从断点续跑。

设计沿用 task_kernel/graph.py 的约定：
- state 只放紧凑业务字段（items 的 content 已被 fetch_max_chars 截断）
- thread_id = job_id；恢复 = 同 thread_id 传 None
- 节点复用 research.py 的纯函数（_plan/_collect/_replan/_synthesize/
  _materials_thin）——编排层换了，业务能力零改动
- 交付不在图里：正常路径由 AsyncJob runner 汇报（P0 口吻播报），
  崩溃恢复路径由 recover() 直接补交付（record_outcome + _voice_outcome）

开关：[deep_research] engine = "legacy"（默认）| "langgraph"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TypedDict

from junjun_core.observability import get_logger
from junjun_skills.plugins.async_task import research

logger = get_logger("async_task.research_graph")


class ResearchState(TypedDict, total=False):
    topic: str
    chat_id: str
    job_id: str
    queries: list       # 当前轮查询
    items: list         # [{title,url,snippet,content}]（content 已截断）
    round_no: int
    report: str
    error: str


def _max_rounds() -> int:
    return int(research._cfg().get("max_rounds", 2))


def _thread_cfg(job_id: str) -> dict:
    # plan→collect→(reflect→collect)→synthesize：每轮 2-3 节点，余量放宽
    return {"configurable": {"thread_id": job_id},
            "recursion_limit": 4 * _max_rounds() + 10}


# ---------------------------------------------------------------------------
# 节点（build_graph 闭包注入依赖，测试可换桩；生产全 None 走默认）
# ---------------------------------------------------------------------------

def _make_nodes(plan_model=None, synth_model=None, search=None, fetch=None):
    async def plan_node(state: ResearchState) -> dict:
        queries = await research._plan(state["topic"], plan_model)
        logger.info(f"深度研究 [{state['job_id']}] 拆出 {len(queries)} 个查询（图引擎）")
        return {"queries": queries, "round_no": 1}

    async def collect_node(state: ResearchState) -> dict:
        new_items = await research._collect(state["queries"],
                                            search=search, fetch=fetch)
        items = list(state.get("items") or [])
        seen = {it["url"] for it in items}
        items += [it for it in new_items if it["url"] not in seen]
        return {"items": items}

    async def reflect_node(state: ResearchState) -> dict:
        got = sum(1 for it in (state.get("items") or []) if it.get("content"))
        new_q = await research._replan(state["topic"], state["queries"],
                                       got, plan_model)
        round_no = int(state.get("round_no", 1)) + 1
        if not new_q:
            # 规划器想不出新角度：直接把 round_no 顶满，条件边去综述
            return {"round_no": _max_rounds()}
        logger.info(f"深度研究 [{state['job_id']}] 材料薄，反思改写再搜 "
                    f"{len(new_q)} 个查询（第 {round_no} 轮）")
        return {"queries": new_q, "round_no": round_no}

    async def synthesize_node(state: ResearchState) -> dict:
        items = state.get("items") or []
        if not items:
            return {"error": "所有搜索引擎都没查到东西，换个说法试试"}
        report = await research._synthesize(state["topic"], items, synth_model)
        return {"report": report}

    def route_after_collect(state: ResearchState) -> str:
        if (int(state.get("round_no", 1)) < _max_rounds()
                and research._materials_thin(state.get("items") or [])):
            return "reflect"
        return "synthesize"

    return plan_node, collect_node, reflect_node, synthesize_node, route_after_collect


def build_graph(checkpointer, *, plan_model=None, synth_model=None,
                search=None, fetch=None):
    from langgraph.graph import END, START, StateGraph
    plan_node, collect_node, reflect_node, synthesize_node, route = _make_nodes(
        plan_model, synth_model, search, fetch)
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("collect", collect_node)
    g.add_node("reflect", reflect_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "collect")
    g.add_conditional_edges("collect", route,
                            {"reflect": "reflect", "synthesize": "synthesize"})
    g.add_edge("reflect", "collect")
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 运行时：执行 / 注册表 / 启动恢复补交付
# ---------------------------------------------------------------------------

_persist_dir: Optional[Path] = None
_graph = None
_recovered = False


def configure(persist_dir) -> None:
    """生产启动挂钩（run_junjun）。测试勿调（直接 build_graph(MemorySaver)）。"""
    global _persist_dir
    _persist_dir = Path(persist_dir)


async def _ensure_graph():
    global _graph
    if _graph is None:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        if _persist_dir:
            _persist_dir.mkdir(parents=True, exist_ok=True)
            db = _persist_dir / "research.db"
        else:
            db = Path(":memory:")
        conn = await aiosqlite.connect(str(db))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        _graph = build_graph(saver)
    return _graph


def _registry_path() -> Optional[Path]:
    return (_persist_dir / "active_research.json") if _persist_dir else None


def _registry_load() -> dict:
    p = _registry_path()
    if not p or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _registry_save(d: dict) -> None:
    p = _registry_path()
    if not p:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.warning(f"研究任务注册表落盘失败（忽略）: {e}")


def registry_add(job_id: str, chat_id: str, topic: str) -> None:
    d = _registry_load()
    d[job_id] = {"chat_id": chat_id, "topic": topic[:80]}
    _registry_save(d)


def registry_remove(job_id: str) -> None:
    d = _registry_load()
    if job_id in d:
        d.pop(job_id)
        _registry_save(d)


async def run(job, topic: str) -> str:
    """图引擎执行入口（research.deep_research_handler 分流到这里）。
    返回报告；失败抛 RuntimeError——与 legacy 路径的 runner 语义完全一致。"""
    graph = await _ensure_graph()
    registry_add(job.job_id, job.chat_id, topic)
    state = await graph.ainvoke(
        {"topic": topic, "chat_id": job.chat_id, "job_id": job.job_id},
        _thread_cfg(job.job_id))
    # 先摘注册表再交付：崩溃窗口内宁可按「重启中断」如实上报，也不重复播报
    registry_remove(job.job_id)
    if state.get("error"):
        raise RuntimeError(state["error"])
    return state["report"]


async def _deliver_recovered(job_id: str, chat_id: str, topic: str,
                             report: str) -> None:
    """崩溃恢复后的补交付：结局登记 + Agent 口吻播报（复用 P0 通道）。"""
    from junjun_agent.tasks import task_manager
    task_manager._record_outcome(chat_id, "deep_research", "done",
                                 f"{topic[:30]}：研究完成（重启后续跑）", said="")
    try:
        await task_manager._voice_outcome(chat_id, "deep_research", True,
                                          report, 0.0, topic)
    except Exception as e:
        logger.warning(f"恢复任务播报失败（结局已登记）: {type(e).__name__}: {e}")


async def recover() -> None:
    """启动恢复：注册表里的研究任务断点续跑（input None）并完成补交付。"""
    global _recovered
    if _recovered:
        return
    _recovered = True
    registry = _registry_load()
    if not registry:
        return
    from langgraph.errors import EmptyInputError
    graph = await _ensure_graph()
    for job_id, info in list(registry.items()):
        try:
            state = await graph.ainvoke(None, _thread_cfg(job_id))
        except EmptyInputError:
            # 已完成但注册表没来得及摘：读快照补交付
            snap = await graph.aget_state(_thread_cfg(job_id))
            state = snap.values or {}
        except Exception as e:
            logger.warning(f"研究任务 {job_id} 断点恢复失败: {type(e).__name__}: {e}")
            continue
        registry_remove(job_id)
        if state.get("report"):
            await _deliver_recovered(job_id, info.get("chat_id", ""),
                                     info.get("topic", ""), state["report"])
            logger.info(f"研究任务 {job_id} 已续跑完成并补交付")
        else:
            from junjun_agent.tasks import task_manager
            task_manager._record_outcome(
                info.get("chat_id", ""), "deep_research", "failed",
                f"{info.get('topic', '')[:30]}：{state.get('error', '续跑未完成')}")
            logger.warning(f"研究任务 {job_id} 续跑未出报告: {state.get('error')}")
