"""热点日报的 LangGraph 管线：选题→深研→写稿→人审→发空间。

设计（LangGraph 迁移二期，与 task_kernel/research_graph 同一套约定）：
- state 只放紧凑业务字段；thread_id = report_id（"dr-YYYY-MM-DD"，一天一单）
- 节点复用现成能力：news 60s / topic_finder RSS 取素材、research.py 纯函数
  深研、junzone.publish_feed 发布——本图只做编排与人审
- 人审 = interrupt + 私聊管理员，回「发」放行 /「算了」丢弃，
  超时默认【不发】（对外发布，保守方向）
- 崩溃断点续跑：sqlite checkpoint + active_reports.json 注册表，
  启动 recover() 续跑；停在人审的重新通知管理员

交付语义：发出去（tid）/ 跳过（skip_reason）/ 失败（error）三选一，
都写 history.json 防同日重发、选题防重复。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional, TypedDict

from junjun_core.observability import get_logger

logger = get_logger("plugin.daily_report.graph")


class ReportState(TypedDict, total=False):
    report_id: str
    date: str
    news_titles: list      # 素材标题池
    topic: str             # 选定主题
    research_report: str   # 深研综述
    draft: str             # 说说成稿
    tid: str               # 发布成功（QZone feed id）
    skip_reason: str       # 主动跳过（素材空/没选题/没材料/被驳回/超时）
    error: str             # 失败


def _thread_cfg(report_id: str) -> dict:
    # 6 节点线性，余量放宽；测试用更小的 limit 模拟崩溃
    return {"configurable": {"thread_id": report_id}, "recursion_limit": 20}


# ---------------------------------------------------------------------------
# 节点（build_graph 闭包注入依赖，测试换桩；生产全 None 走 tools.py 默认）
# ---------------------------------------------------------------------------

def _make_nodes(deps, END):
    async def gather_node(state: ReportState) -> dict:
        titles = await deps["gather"]()
        logger.info(f"热点日报 [{state['report_id']}] 抓到 {len(titles)} 条素材")
        if not titles:
            return {"news_titles": [], "skip_reason": "没抓到热点素材"}
        return {"news_titles": titles}

    async def pick_node(state: ReportState) -> dict:
        topic = await deps["pick"](state["news_titles"],
                                   deps["recent_topics"]())
        if not topic:
            return {"skip_reason": "今天没有值得深研的热点"}
        logger.info(f"热点日报 [{state['report_id']}] 选定主题: {topic[:40]}")
        return {"topic": topic}

    async def research_node(state: ReportState) -> dict:
        from junjun_skills.plugins.async_task import research
        topic = state["topic"]
        queries = await research._plan(topic, deps["plan_model"])
        items = await research._collect(queries, search=deps["search"],
                                        fetch=deps["fetch"])
        if research._materials_thin(items) and int(research._cfg().get("max_rounds", 2)) > 1:
            new_q = await research._replan(topic, queries, len(items),
                                           deps["plan_model"])
            if new_q:
                seen = {it["url"] for it in items}
                more = await research._collect(new_q, search=deps["search"],
                                               fetch=deps["fetch"])
                items += [it for it in more if it["url"] not in seen]
        if not items:
            return {"skip_reason": "深研没查到材料"}
        report = await research._synthesize(topic, items, deps["synth_model"])
        return {"research_report": report}

    async def write_node(state: ReportState) -> dict:
        draft = await deps["write"](state["topic"], state["research_report"])
        if not draft:
            return {"error": "写稿模型返回空"}
        return {"draft": draft}

    async def approval_node(state: ReportState) -> dict:
        from langgraph.types import interrupt
        approved = interrupt({
            "kind": "daily_report",
            "report_id": state["report_id"],
            "topic": state["topic"],
            "draft": state["draft"],
        })
        if approved:
            return {}
        return {"skip_reason": "管理员没放行（或超时默认丢弃）"}

    async def publish_node(state: ReportState) -> dict:
        tid = await deps["publish"](state["draft"])
        if not tid:
            return {"error": "空间发布失败（登录态/网络）"}
        logger.info(f"热点日报 [{state['report_id']}] 已发布 tid={tid}")
        return {"tid": str(tid)}

    def _has_titles(state: ReportState) -> str:
        return "pick" if state.get("news_titles") else END

    def _has_topic(state: ReportState) -> str:
        return "research" if state.get("topic") else END

    def _has_report(state: ReportState) -> str:
        return "write" if state.get("research_report") else END

    def _has_draft(state: ReportState) -> str:
        return "approval" if state.get("draft") else END

    def _approved(state: ReportState) -> str:
        # 驳回/超时（approval 节点写入 skip_reason）直接完结，不进发布
        return END if state.get("skip_reason") else "publish"

    return (gather_node, pick_node, research_node, write_node,
            approval_node, publish_node,
            _has_titles, _has_topic, _has_report, _has_draft, _approved)


def default_deps() -> dict:
    """生产默认依赖（late import 防循环；测试用 build_graph 注入桩）。"""
    from junjun_skills.plugins.daily_report import tools
    return {
        "gather": tools.gather_materials,
        "pick": tools.pick_topic,
        "recent_topics": tools.recent_topics,
        "write": tools.write_draft,
        "publish": tools.publish_draft,
        "plan_model": None, "synth_model": None,
        "search": None, "fetch": None,
    }


def build_graph(checkpointer, deps: Optional[dict] = None):
    from langgraph.graph import END, START, StateGraph
    d = {**default_deps(), **(deps or {})}
    (gather_node, pick_node, research_node, write_node,
     approval_node, publish_node,
     _has_titles, _has_topic, _has_report, _has_draft, _approved) = _make_nodes(d, END)
    g = StateGraph(ReportState)
    g.add_node("gather", gather_node)
    g.add_node("pick", pick_node)
    g.add_node("research", research_node)
    g.add_node("write", write_node)
    g.add_node("approval", approval_node)
    g.add_node("publish", publish_node)
    g.add_edge(START, "gather")
    g.add_conditional_edges("gather", _has_titles, {"pick": "pick", END: END})
    g.add_conditional_edges("pick", _has_topic, {"research": "research", END: END})
    g.add_conditional_edges("research", _has_report, {"write": "write", END: END})
    g.add_conditional_edges("write", _has_draft, {"approval": "approval", END: END})
    g.add_conditional_edges("approval", _approved, {"publish": "publish", END: END})
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 运行时：执行 / 人审 / 注册表 / 启动恢复
# ---------------------------------------------------------------------------

_persist_dir: Optional[Path] = None
_graph = None
_recovered = False
_pending: dict = {}   # report_id -> {topic, draft, timeout_task}


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
            db = _persist_dir / "daily_report.db"
        else:
            db = Path(":memory:")
        conn = await aiosqlite.connect(str(db))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        _graph = build_graph(saver)
    return _graph


def _registry_path() -> Optional[Path]:
    return (_persist_dir / "active_reports.json") if _persist_dir else None


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
        logger.warning(f"日报注册表落盘失败（忽略）: {e}")


def registry_add(report_id: str, date: str) -> None:
    d = _registry_load()
    d[report_id] = {"date": date}
    _registry_save(d)


def registry_remove(report_id: str) -> None:
    d = _registry_load()
    if report_id in d:
        d.pop(report_id)
        _registry_save(d)


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("daily_report", {}) or {}
    except Exception:
        return {}


async def _notify_admin(report_id: str, topic: str, draft: str) -> None:
    from junjun_core.security import notify_admin
    text = (f"【热点日报审批】今天想聊：{topic}\n\n{draft}\n\n"
            f"回「发」发到空间，回「算了」今天不发。10 分钟没回默认不发。")
    try:
        if not await notify_admin(text):
            logger.warning(f"日报审批通知未送达（未配置 ADMIN_QQ？），"
                           f"超时将默认不发: {report_id}")
    except Exception as e:
        logger.warning(f"日报审批通知管理员失败: {type(e).__name__}: {e}")


def _arm_timeout(report_id: str) -> None:
    timeout = float(_cfg().get("approval_timeout_seconds", 600))

    async def _watch():
        await asyncio.sleep(timeout)
        if report_id in _pending:
            logger.info(f"日报审批超时（{timeout:.0f}s 无回复），默认不发: {report_id}")
            await resume(report_id, False)

    _pending[report_id]["timeout_task"] = asyncio.create_task(_watch())


async def _after_run(report_id: str, state: dict) -> dict:
    """ainvoke 返回后的统一收尾：人审中断 vs 完结（发布/跳过/失败）。"""
    if "__interrupt__" in state:
        payload = (state["__interrupt__"][0].value
                   if state["__interrupt__"] else {})
        _pending[report_id] = {"topic": payload.get("topic", ""),
                               "draft": payload.get("draft", "")}
        await _notify_admin(report_id, payload.get("topic", ""),
                            payload.get("draft", ""))
        _arm_timeout(report_id)
        logger.info(f"热点日报 [{report_id}] 等待管理员审批")
        return state
    registry_remove(report_id)
    if state.get("tid"):
        _record_history(report_id, state)
    elif state.get("skip_reason"):
        logger.info(f"热点日报 [{report_id}] 跳过: {state['skip_reason']}")
    elif state.get("error"):
        logger.warning(f"热点日报 [{report_id}] 失败: {state['error']}")
    return state


def _record_history(report_id: str, state: dict) -> None:
    """发布成功才进历史（防同日重发/选题去重）；跳过/失败不进，明天还能选同题。"""
    try:
        from junjun_skills.plugins.daily_report import tools
        tools.record_history(state.get("date", ""), state.get("topic", ""),
                             state.get("tid", ""))
    except Exception as e:
        logger.warning(f"日报历史落盘失败（忽略）: {e}")


async def run(report_id: str, date: str) -> dict:
    """执行一单日报。正常路径：跑到人审中断（通知管理员）或直接完结。"""
    graph = await _ensure_graph()
    registry_add(report_id, date)
    try:
        state = await graph.ainvoke({"report_id": report_id, "date": date},
                                    _thread_cfg(report_id))
    except Exception as e:
        registry_remove(report_id)
        logger.warning(f"热点日报 [{report_id}] 执行异常: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}
    return await _after_run(report_id, state)


async def resume(report_id: str, approved: bool) -> None:
    """管理员审批结果回灌（Command(resume=...)）。"""
    pend = _pending.pop(report_id, None)
    if pend and pend.get("timeout_task"):
        pend["timeout_task"].cancel()
    graph = await _ensure_graph()
    from langgraph.types import Command
    try:
        state = await graph.ainvoke(Command(resume=approved),
                                    _thread_cfg(report_id))
    except Exception as e:
        registry_remove(report_id)
        logger.warning(f"热点日报 [{report_id}] 审批续跑异常: {type(e).__name__}: {e}")
        return
    await _after_run(report_id, state)


async def recover() -> None:
    """启动恢复：注册表里的日报断点续跑；停在人审的重建待审批并重新通知。"""
    global _recovered
    if _recovered:
        return
    _recovered = True
    registry = _registry_load()
    if not registry:
        return
    from langgraph.errors import EmptyInputError
    graph = await _ensure_graph()
    for report_id, info in list(registry.items()):
        try:
            state = await graph.ainvoke(None, _thread_cfg(report_id))
        except EmptyInputError:
            # 已完结但注册表没来得及摘：读快照走正常收尾
            snap = await graph.aget_state(_thread_cfg(report_id))
            state = snap.values or {}
        except Exception as e:
            logger.warning(f"日报 {report_id} 断点恢复失败: {type(e).__name__}: {e}")
            continue
        # sqlite 续跑到人审会重新 interrupt（__interrupt__ 在 state 里），
        # 已完结的走正常收尾——统一交给 _after_run（重建 pending/摘注册表）
        await _after_run(report_id, state)


# ---------------------------------------------------------------------------
# processor 入站钩子：管理员的「发/算了」
# ---------------------------------------------------------------------------

_APPROVE_WORDS = {"发": True, "算了": False}


async def approval_hook(session, meta) -> bool:
    """True=已消费（不进决策队列）。管理员本人 + 精确审批词 + 有待审批日报
    才拦截——与 task_kernel 审批同一套词，各自只在自己的 pending 非空时接单。"""
    from junjun_core.security import is_admin
    if not is_admin(meta.user_id):
        return False
    decision = _APPROVE_WORDS.get((meta.text or "").strip())
    if decision is None or not _pending:
        return False
    report_id = next(iter(_pending))  # FIFO：一次只批最早一单
    info = _pending.get(report_id, {})
    asyncio.create_task(resume(report_id, decision))
    ack = "好，这就发到空间。" if decision else "行，今天这条不发了。"
    try:
        from junjun_agent.outbound import send_proactive
        from junjun_core.contracts import ReplySegment
        await send_proactive(session.chat_id, [ReplySegment(type="text", data=ack)],
                             source="daily_report", remember=False)
    except Exception:
        pass
    logger.info(f"管理员审批 {'放行' if decision else '丢弃'}日报: "
                f"{report_id} {info.get('topic', '')[:40]}")
    return True
