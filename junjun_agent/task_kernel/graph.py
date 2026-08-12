"""TaskKernel LangGraph 引擎：StateGraph + checkpointer 的崩溃续跑与人审中断。

设计：docs/LangGraph迁移设计_TaskKernel_2026-08-09.md。要点：
- state 只放紧凑业务字段（plan dict，step result ≤500 字），不放 messages
  通道——规避 langgraph#7714 的 +37.8% token 膨胀。
- thread_id = plan_id；断点恢复 = 同 thread_id + input 传 None
  （传了 input 会从头跑——调研高频坑）。
- 人审 = approval 节点 interrupt 挂起 + 管理员私聊「发/算了」
  Command(resume=...) 继续；超时默认跳过（宁保守不放行）。
- 步骤执行件（_run_step/_report 等）复用 executor.kernel 的无状态方法，
  与 legacy 引擎同一实现，行为不漂移。
- 规划不在图里：try_submit 要先同步拿到 plan 才能决定接单还是回退对话通道，
  图从 execute 开始（与设计文档 §3 的偏差，刻意为之）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional, TypedDict

from junjun_core.observability import get_logger
from junjun_agent.task_kernel.plan import TaskPlan

logger = get_logger("task_kernel.graph")


class KernelState(TypedDict, total=False):
    plan: dict          # TaskPlan.to_dict()（steps 内联，紧凑）
    phase: str          # execute / decide / replan / approval / report
    awaiting: str       # 待审批步骤 id（""=无）
    replan_for: str     # 待局部重规划的失败步骤 id


def _cfg() -> dict:
    from junjun_agent.task_kernel.executor import _cfg as _ecfg
    return _ecfg()


def _recursion_limit() -> int:
    """每步最多 execute+decide 两轮（重试一次）×2 superstep，加余量。"""
    return 4 * int(_cfg().get("max_steps", 6)) + 10


def _thread_cfg(plan_id: str) -> dict:
    return {"configurable": {"thread_id": plan_id},
            "recursion_limit": _recursion_limit()}


# ---------------------------------------------------------------------------
# 节点（纯函数：输入 state，返回 partial state；不碰实例状态）
# ---------------------------------------------------------------------------

async def execute_node(state: KernelState) -> dict:
    """跑一轮就绪步骤（并行）；human 门未批准的步骤跳过，留给 decide 路由去审批。"""
    from junjun_agent.task_kernel.executor import kernel
    plan = TaskPlan.from_dict(state["plan"])
    if plan.deadline_ts and time.time() > plan.deadline_ts:
        plan.state, plan.note = "failed", "超过时限"
        return {"plan": plan.to_dict(), "phase": "decide"}
    ready = plan.ready_steps()
    auto = [s for s in ready if not (s.verify == "human" and not s.approved)]
    if auto:
        await asyncio.gather(*(kernel._run_step(plan, s) for s in auto),
                             return_exceptions=True)
    return {"plan": plan.to_dict(), "phase": "decide"}


async def decide_node(state: KernelState) -> dict:
    """失败决策（retry/replan/abort）+ 人审路由——legacy while 循环体的图化。"""
    plan = TaskPlan.from_dict(state["plan"])
    if plan.state in ("done", "failed"):
        return {"plan": plan.to_dict(), "phase": "report"}

    max_replans = int(_cfg().get("max_replans", 1))
    failed = [s for s in plan.steps if s.status == "failed"]
    # 验证失败计数（audit 指标）：区别于工具/网络类失败；重试重置后 error 清空
    # 不会重复计，与 legacy 语义一致
    plan.verify_failures += sum(
        1 for s in failed if s.error.startswith(("验证未通过", "验收不通过")))
    replan_for = ""
    for s in failed:
        if plan.attempts.get(s.id, 0) < 2:
            logger.info(f"步骤 {s.id} 失败（{s.error[:60]}），重试一次")
            s.status, s.error = "pending", ""
        elif plan.replans < max_replans and not replan_for:
            replan_for = s.id
        else:
            plan.state = "failed"
            plan.note = f"步骤「{s.desc[:30]}」失败：{s.error[:100]}"
    if plan.state == "failed":
        return {"plan": plan.to_dict(), "phase": "report", "replan_for": ""}
    if replan_for:
        return {"plan": plan.to_dict(), "phase": "replan", "replan_for": replan_for}

    ready = plan.ready_steps()
    gated = [s for s in ready if s.verify == "human" and not s.approved]
    auto = [s for s in ready if s not in gated]
    if auto:
        return {"plan": plan.to_dict(), "phase": "execute", "replan_for": ""}
    if gated:
        return {"plan": plan.to_dict(), "phase": "approval",
                "awaiting": gated[0].id, "replan_for": ""}
    if all(s.status in ("done", "skipped") for s in plan.steps):
        plan.state = "done"
    elif any(s.status == "pending" for s in plan.steps):
        plan.state, plan.note = "failed", "步骤图无法推进（依赖断裂）"
    return {"plan": plan.to_dict(), "phase": "report", "replan_for": ""}


async def replan_node(state: KernelState) -> dict:
    """局部重规划：只重写未执行的剩余步骤（planner.revise_remaining，thinker 槽）。"""
    from junjun_agent.task_kernel.planner import revise_remaining
    plan = TaskPlan.from_dict(state["plan"])
    sid = state.get("replan_for") or ""
    step = next((s for s in plan.steps if s.id == sid), None)
    plan.replans += 1
    logger.info(f"步骤 {sid} 连续失败，局部重规划（第 {plan.replans} 次）")
    new_steps = None
    try:
        if step is not None:
            new_steps = await revise_remaining(plan, step.desc, step.error)
    except Exception as e:
        logger.warning(f"局部重规划异常: {type(e).__name__}: {e}")
    if new_steps:
        plan.steps = [x for x in plan.steps if x.status == "done"] + list(new_steps)
        return {"plan": plan.to_dict(), "phase": "decide", "replan_for": ""}
    if step is not None:
        step.status = "failed"
    plan.state = "failed"
    plan.note = f"步骤「{(step.desc if step else sid)[:30]}」失败且无法重规划"
    return {"plan": plan.to_dict(), "phase": "report", "replan_for": ""}


async def approval_node(state: KernelState) -> dict:
    """人审门：interrupt 挂起；Command(resume=True/False) 恢复。重启后重新挂起，
    runner 侧负责重新通知管理员。"""
    from langgraph.types import interrupt
    plan = TaskPlan.from_dict(state["plan"])
    sid = state.get("awaiting") or ""
    step = next((s for s in plan.steps if s.id == sid), None)
    approved = interrupt({
        "kind": "task_approval",
        "plan_id": plan.plan_id,
        "chat_id": plan.chat_id,
        "goal": plan.goal[:80],
        "step": {"id": sid,
                 "action": step.action if step else "",
                 "desc": step.desc if step else ""},
    })
    if step is not None:
        if approved:
            step.approved = True
            logger.info(f"[{plan.chat_id}] 步骤 {sid} 管理员批准，放行执行")
        else:
            step.status = "skipped"
            logger.info(f"[{plan.chat_id}] 步骤 {sid} 跳过（管理员否决/审批超时）")
    return {"plan": plan.to_dict(), "phase": "decide", "awaiting": ""}


async def report_node(state: KernelState) -> dict:
    """终态汇报（口吻合成复用 legacy）+ 活动注册表清理。"""
    from junjun_agent.task_kernel.executor import kernel
    plan = TaskPlan.from_dict(state["plan"])
    if plan.state not in ("done", "failed"):
        plan.state = "failed"
        plan.note = plan.note or "流程异常终止"
    try:
        await kernel._report(plan)
    finally:
        runner.registry_remove(plan.plan_id)
    return {"plan": plan.to_dict(), "phase": "end"}


def _route(state: KernelState) -> str:
    return {"execute": "execute", "replan": "replan",
            "approval": "approval"}.get(state.get("phase", ""), "report")


def build_graph(checkpointer):
    """编译任务图。测试传 MemorySaver；生产由 GraphRunner 传 AsyncSqliteSaver。"""
    from langgraph.graph import END, START, StateGraph
    g = StateGraph(KernelState)
    g.add_node("execute", execute_node)
    g.add_node("decide", decide_node)
    g.add_node("replan", replan_node)
    g.add_node("approval", approval_node)
    g.add_node("report", report_node)
    g.add_edge(START, "execute")
    g.add_edge("execute", "decide")
    g.add_conditional_edges("decide", _route, {
        "execute": "execute", "replan": "replan",
        "approval": "approval", "report": "report",
    })
    g.add_edge("replan", "decide")
    g.add_edge("approval", "decide")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 运行时：提交 / 审批恢复 / 超时 / 注册表 / 启动断点恢复
# ---------------------------------------------------------------------------

class GraphRunner:
    def __init__(self) -> None:
        self._graph = None
        self._persist_dir: Optional[Path] = None
        self._pending: dict = {}        # plan_id -> {goal, desc, timeout_task}
        self._recovered = False

    # ---- 装配 ----

    def configure(self, persist_dir) -> None:
        self._persist_dir = Path(persist_dir)

    async def _ensure_graph(self):
        if self._graph is None:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            if self._persist_dir:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                db = self._persist_dir / "kernel.db"
            else:
                db = Path(":memory:")
            conn = await aiosqlite.connect(str(db))
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            self._graph = build_graph(saver)
        return self._graph

    # ---- 活动任务注册表（崩溃续跑的索引；checkpoint 本身没法按「未完成」枚举） ----

    @property
    def _registry_path(self) -> Optional[Path]:
        return (self._persist_dir / "active_plans.json") if self._persist_dir else None

    def _registry_load(self) -> dict:
        p = self._registry_path
        if not p or not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _registry_save(self, d: dict) -> None:
        p = self._registry_path
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:
            logger.warning(f"活动任务注册表落盘失败（忽略）: {e}")

    def registry_add(self, plan: TaskPlan) -> None:
        d = self._registry_load()
        d[plan.plan_id] = {"chat_id": plan.chat_id, "goal": plan.goal[:80]}
        self._registry_save(d)

    def registry_remove(self, plan_id: str) -> None:
        d = self._registry_load()
        if plan_id in d:
            d.pop(plan_id)
            self._registry_save(d)

    # ---- 提交与恢复 ----

    async def submit(self, plan: TaskPlan) -> None:
        graph = await self._ensure_graph()
        self.registry_add(plan)
        try:
            state = await graph.ainvoke(
                {"plan": plan.to_dict(), "phase": "execute",
                 "awaiting": "", "replan_for": ""},
                _thread_cfg(plan.plan_id))
        except Exception as e:
            # 图级异常（如 recursion_limit 烧穿）：照实汇报失败，不装死
            logger.warning(f"[{plan.chat_id}] 任务图异常终止: {type(e).__name__}: {e}")
            plan.state, plan.note = "failed", f"{type(e).__name__}: {e}"
            from junjun_agent.task_kernel.executor import kernel
            await kernel._report(plan)
            self.registry_remove(plan.plan_id)
            return
        await self._after_run(plan, state)

    async def resume(self, plan_id: str, approved: bool) -> None:
        """审批决议恢复执行（Command(resume=...)）。"""
        from langgraph.types import Command
        pend = self._pending.pop(plan_id, None)
        if pend and pend.get("timeout_task"):
            task = pend["timeout_task"]
            # 看门狗超时路径是 _watch 自己调 resume——cancel 自己会让
            # CancelledError 在下一个 await 把 resume 整体炸掉，计划永远卡在
            # interrupt（生产后果：管理员 10 分钟没回 -> 任务挂死无汇报；
            # 2026-08-12 eval approve-timeout 实锤，单测只断言 pending 弹出
            # 没断言终态所以一直没抓到）。
            if task is not asyncio.current_task():
                task.cancel()
        graph = await self._ensure_graph()
        try:
            state = await graph.ainvoke(Command(resume=approved),
                                        _thread_cfg(plan_id))
        except Exception as e:
            logger.warning(f"审批恢复执行异常: {type(e).__name__}: {e}")
            return
        await self._after_run(TaskPlan.from_dict(state["plan"]), state)

    async def recover(self) -> None:
        """启动恢复：注册表里的活动任务逐个断点续跑——input 必须传 None，
        传了 input 会从头跑（调研高频坑）。审批挂起的任务传 None 会抛
        EmptyInputError（interrupt 只认 Command(resume=...)），读快照重建
        待审批状态并重新通知管理员。"""
        if self._recovered:
            return
        self._recovered = True
        registry = self._registry_load()
        if not registry:
            return
        from langgraph.errors import EmptyInputError
        graph = await self._ensure_graph()
        for plan_id in list(registry):
            try:
                state = await graph.ainvoke(None, _thread_cfg(plan_id))
            except EmptyInputError:
                await self._restore_pending_approval(graph, plan_id)
                continue
            except Exception as e:
                logger.warning(f"任务 {plan_id} 断点恢复失败: {type(e).__name__}: {e}")
                continue
            plan = TaskPlan.from_dict(state["plan"])
            await self._after_run(plan, state)
            logger.info(f"任务 {plan_id} 已从断点恢复（{registry[plan_id].get('goal', '')[:30]}）")

    async def _restore_pending_approval(self, graph, plan_id: str) -> None:
        """审批挂起任务的启动恢复：从 checkpoint 快照重建待审批 + 重新通知。"""
        try:
            snap = await graph.aget_state(_thread_cfg(plan_id))
            values = snap.values or {}
            awaiting = values.get("awaiting") or ""
            if not awaiting:
                logger.warning(f"任务 {plan_id} 恢复时空输入但无待审批步骤，保持挂起待人工")
                return
            plan = TaskPlan.from_dict(values["plan"])
            step = next((s for s in plan.steps if s.id == awaiting), None)
            self._pending[plan_id] = {
                "goal": plan.goal[:80], "desc": step.desc if step else "",
            }
            await self._notify_admin(plan, {
                "goal": plan.goal[:80],
                "step": {"id": awaiting,
                         "action": step.action if step else "",
                         "desc": step.desc if step else ""},
            })
            self._arm_timeout(plan_id)
            logger.info(f"任务 {plan_id} 的待审批已从断点恢复（重新通知管理员）")
        except Exception as e:
            logger.warning(f"任务 {plan_id} 待审批恢复失败: {type(e).__name__}: {e}")

    # ---- 审批 ----

    async def _after_run(self, plan: TaskPlan, state: dict) -> None:
        """图一次运行的收尾：挂起（__interrupt__）→ 通知管理员 + 超时看门狗；
        终态 → 注册表已在 report 节点清理（这里幂等兜底）。"""
        intr = state.get("__interrupt__") if isinstance(state, dict) else None
        if intr:
            value = getattr(intr[0], "value", None) or {}
            step = value.get("step", {})
            self._pending[plan.plan_id] = {
                "goal": value.get("goal", ""), "desc": step.get("desc", ""),
            }
            await self._notify_admin(plan, value)
            self._arm_timeout(plan.plan_id)
        else:
            self.registry_remove(plan.plan_id)

    async def _notify_admin(self, plan: TaskPlan, payload: dict) -> None:
        from junjun_core.security import notify_admin
        step = payload.get("step", {})
        text = (f"【任务审批】{payload.get('goal', '')}\n"
                f"下一步要做：{step.get('desc', '')}（{step.get('action', '')}）\n"
                f"回「发」放行，回「算了」跳过。10 分钟没回默认跳过。")
        try:
            if not await notify_admin(text):
                logger.warning(f"审批通知未送达（未配置 ADMIN_QQ？），"
                               f"将走超时默认跳过: {plan.plan_id}")
        except Exception as e:
            logger.warning(f"审批通知管理员失败: {type(e).__name__}: {e}")

    def _arm_timeout(self, plan_id: str) -> None:
        timeout = float(_cfg().get("approval_timeout_seconds", 600))

        async def _watch():
            await asyncio.sleep(timeout)
            if plan_id in self._pending:
                logger.info(f"审批超时（{timeout:.0f}s 无回复），默认跳过: {plan_id}")
                await self.resume(plan_id, False)

        self._pending[plan_id]["timeout_task"] = asyncio.create_task(_watch())

    @property
    def pending_approvals(self) -> dict:
        return self._pending


runner = GraphRunner()


# ---------------------------------------------------------------------------
# processor 入站钩子：管理员的「发/算了」最优先消费
# ---------------------------------------------------------------------------

_APPROVE_WORDS = {"发": True, "算了": False}


async def approval_hook(session, meta) -> bool:
    """True=已消费（不进决策队列）。只在 LangGraph 引擎 + 管理员本人 +
    精确命中审批词 + 有待审批任务时拦截——日常句子零误伤。"""
    from junjun_agent.task_kernel.executor import engine
    if engine() != "langgraph":
        return False
    from junjun_core.security import is_admin
    if not is_admin(meta.user_id):
        return False
    decision = _APPROVE_WORDS.get((meta.text or "").strip())
    if decision is None or not runner.pending_approvals:
        return False
    plan_id = next(iter(runner.pending_approvals))  # FIFO：一次只批最早一单
    info = runner.pending_approvals.get(plan_id, {})
    asyncio.create_task(runner.resume(plan_id, decision))
    ack = "好，这步放行。" if decision else "行，这步跳过。"
    try:
        from junjun_agent.outbound import send_proactive
        from junjun_core.contracts import ReplySegment
        await send_proactive(session.chat_id, [ReplySegment(type="text", data=ack)],
                             source="task_kernel", remember=False)
    except Exception:
        pass
    logger.info(f"管理员审批 {'放行' if decision else '跳过'}: "
                f"{plan_id} {info.get('desc', '')[:40]}")
    return True
