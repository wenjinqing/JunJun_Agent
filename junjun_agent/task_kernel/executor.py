"""TaskKernel 执行器：步骤图的代码侧状态机（方案 §4.3 执行循环）。

设计要点：
- 执行循环由代码驱动——取就绪步骤、验证、失败决策（retry/replan/abort）
  全是代码逻辑，不靠模型自由发挥（12-factor: harness acts, not the model）。
- 成品类步骤（画图/语音）的工具内部自己会走 TaskManager 提交即返回，
  kernel 只把工具返回值当步骤产出——不重复造轮询。
- 汇报走 outbound.send_proactive（统一出站口）；结局登记复用
  task_manager._record_outcome（kind=task_kernel），决策注入自然生效。
- 接单话术是模板池（0 token，走正常出站路径由 processor 发送）；
  最终汇报过 utils 合成 + persona_brief 口吻——能力与人格分层。
- 灰度开关：[task_kernel] enable（默认关）。关掉后 router 命中也回退
  对话通道，等于回到现状。
"""

import asyncio
import json
import random
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, List, Optional

from junjun_core.bg_tasks import fire_and_forget
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger
from junjun_agent.task_kernel.plan import (
    SIDE_EFFECT_ACTIONS, SYNTH_ACTION, Step, TaskPlan,
)

logger = get_logger("task_kernel.executor")

_PERSIST_DIR: Optional[Path] = None   # 生产由 run_junjun 挂接；测试默认 None

# 当前正在执行的步骤是否已过人审（verify=human 且管理员已放行）。
# run_code 等高危工具的放行依据之一：非管理员任务经人审批准后，执行步骤时
# 置 True，工具体据此放行——绕开工具自身权限门的合法路径只有这一条。
_kernel_step_approved: ContextVar[bool] = ContextVar("jj_kernel_step_approved",
                                                     default=False)


def kernel_step_approved() -> bool:
    """当前执行的步骤是否已过人审批准（高危插件工具查询用）。"""
    return _kernel_step_approved.get()

# 接单话术模板池（直发不经 echo guard——保持 8+ 防口头禅）
_ACK_TEMPLATES = [
    "行，这活得拆开弄，我规划一下步骤，做完了喊你。",
    "收到，这个不是一句话能搞定的，我分几步来，弄好向你汇报。",
    "好嘞，我列个步骤慢慢弄，你先忙你的。",
    "嗯这个得花点功夫，我拆成几步来弄，好了叫你。",
    "可以，我规划下流程再动手，结果出来直接发你。",
    "这单接了，步骤有点多，我一条条来，做完了汇报。",
    "明白，我先拆任务再动手，你别等，好了我主动说。",
    "成，这个我来跟，分步弄，有结果了第一时间说。",
]


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("task_kernel", {})
    except Exception:
        return {}


def enabled() -> bool:
    return bool(_cfg().get("enable", False))


def engine() -> str:
    """执行引擎：legacy（手写 while 循环）| langgraph（StateGraph + 断点续跑 + 人审）。"""
    return str(_cfg().get("engine", "legacy")).strip() or "legacy"


def _approval_actions() -> list:
    """强制人审的发布类动作（LangGraph 引擎）；planner 没标也拦。"""
    acts = _cfg().get("approval_actions", ["send_feed"])
    return [str(a) for a in acts] if isinstance(acts, list) else ["send_feed"]


def _apply_approval_gates(plan: TaskPlan) -> None:
    """LangGraph 引擎专属：发布类步骤强制 verify=human（执行前挂起等管理员）。

    Phase 2：run_code 沙箱步骤对非管理员任务同样上人审门（群友要跑代码，
    管理员「发/算了」裁决）；管理员自己的任务直跑——ADMIN_QQ 是信任根。
    """
    gated = set(_approval_actions())
    from junjun_core.security import is_admin
    admin = is_admin(plan.user_id)
    for s in plan.steps:
        if s.verify == "human":
            continue
        if s.action in gated or (s.action == "run_code" and not admin):
            s.verify = "human"


def enable_persistence(dir_path) -> None:
    """生产启动挂钩：计划落盘目录 + 恢复中断计划。测试勿调。"""
    global _PERSIST_DIR
    _PERSIST_DIR = Path(dir_path)
    _restore_interrupted()
    # LangGraph 引擎的活动任务注册表也放这（崩溃续跑靠它 + kernel.db）
    try:
        from junjun_agent.task_kernel import graph as tk_graph
        tk_graph.runner.configure(dir_path)
    except Exception as e:
        logger.warning(f"LangGraph 引擎注册表挂接失败（忽略）: {e}")


def _persist(plan: TaskPlan) -> None:
    if _PERSIST_DIR is None:
        return
    try:
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        (_PERSIST_DIR / f"{plan.plan_id}.json").write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8")
    except Exception as e:
        logger.warning(f"计划落盘失败（忽略）: {e}")


def _restore_interrupted() -> None:
    """重启恢复：v1 不续跑——进行中的计划标 failed 并登记结局，
    让决策注入能如实说「上次那个任务被重启打断了」（不续编幻觉）。"""
    if _PERSIST_DIR is None or not _PERSIST_DIR.exists():
        return
    n = 0
    from junjun_agent.tasks import task_manager
    for f in _PERSIST_DIR.glob("*.json"):
        try:
            plan = TaskPlan.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
        if plan.state in ("planning", "running", "reporting"):
            plan.state = "failed"
            plan.note = "进程重启，任务中断"
            _persist(plan)
            task_manager._record_outcome(plan.chat_id, "task_kernel", "failed",
                                         f"{plan.goal[:30]}：进程重启，任务中断")
            n += 1
    if n:
        logger.info(f"已标记 {n} 个被重启中断的复杂任务")


class TaskKernel:
    """复杂任务的规划-执行-验证-汇报状态机。

    注意：_run_step/_call_tool/_synthesize/_verify/_report 等方法不依赖实例
    状态（self 只是壳），LangGraph 引擎（graph.py）的节点直接复用它们——
    改这些方法时两边同时生效，勿引入实例状态。"""

    def __init__(self) -> None:
        self._plans: Dict[str, TaskPlan] = {}

    # ---------- 入口 ----------

    async def try_submit(self, text: str, *, chat_id: str, user_id: str = "",
                         callbacks=None) -> Optional[str]:
        """路由命中后的接单入口。先回接单话术（0 等待——真人先应声再干活，
        2026-08-13 审查 P1：旧顺序是先沉默规划 10-30s 再蹦接单，全程还堵着
        会话串行队列），规划放后台；规划失败补一句诚实交代（话已说出，
        回退对话通道等于装死）。仅灰度关闭时返回 None。"""
        if not enabled():
            return None
        # 强引用后台任务：裸 create_task 会被 GC 在任意 await 点收走
        # （2026-08-13 生产实锤：规划任务跑 8 分钟后蒸发，无日志无汇报）
        fire_and_forget(
            self._plan_and_run(text, chat_id=chat_id, user_id=user_id,
                               callbacks=callbacks),
            name=f"task-kernel-plan-{chat_id[-12:]}")
        logger.info(f"[{chat_id}] 复杂任务先接单（后台规划）: {text[:40]}")
        return random.choice(_ACK_TEMPLATES)

    async def _plan_and_run(self, text: str, *, chat_id: str, user_id: str,
                            callbacks) -> None:
        """后台规划 + 起执行（try_submit 的下半段）。失败路径必须出声。"""
        from junjun_agent.task_kernel.planner import make_plan
        max_steps = int(_cfg().get("max_steps", 6))
        plan = None
        try:
            plan = await make_plan(text, chat_id=chat_id, user_id=user_id,
                                   max_steps=max_steps, callbacks=callbacks)
        except Exception as e:
            logger.warning(f"[{chat_id}] 后台规划调用异常: {type(e).__name__}: {e}")
        if plan is None:
            logger.info(f"[{chat_id}] 后台规划未接单（planner_none）: {text[:40]}")
            try:
                from junjun_agent.outbound import send_proactive
                await send_proactive(
                    chat_id,
                    [ReplySegment(type="text",
                                  data="我琢磨了一下，这个我好像拆不动——"
                                       "能再具体说说要啥不？或者等我会儿再试。")],
                    source="task_kernel", remember=False)
            except Exception as e:
                logger.warning(f"[{chat_id}] 规划失败交代发送失败: {e}")
            return
        plan.state = "running"
        plan.deadline_ts = time.time() + float(_cfg().get("deadline_minutes", 30)) * 60
        self._plans[plan.plan_id] = plan
        _persist(plan)
        if engine() == "langgraph":
            _apply_approval_gates(plan)
            from junjun_agent.task_kernel import graph as tk_graph
            fire_and_forget(tk_graph.runner.submit(plan),
                            name=f"task-kernel-lg-{plan.plan_id}")
        else:
            fire_and_forget(self._run(plan), name=f"task-kernel-{plan.plan_id}")
        logger.info(f"[{chat_id}] 复杂任务规划完成（{len(plan.steps)} 步，"
                    f"{engine()} 引擎）: {text[:40]}")
        try:
            from junjun_core.observability import trajectory
            trajectory.emit("tk_plan", chat_id, plan_id=plan.plan_id,
                            goal=text[:120], steps=len(plan.steps),
                            engine=engine())
        except Exception:
            pass

    # ---------- 执行循环 ----------

    async def _run(self, plan: TaskPlan) -> None:
        """Langfuse 根 span：plan 为 trace，步骤为 span，终态指标进 metadata（方案 §六）。

        lf 未启用时是 NoopSpan，零成本——测试与 tracing 降级环境行为不变。
        """
        from junjun_core.observability import lf
        with lf.start_span(
            name=f"task_kernel.{plan.chat_id}",
            input={"goal": plan.goal,
                   "steps": [{"id": s.id, "action": s.action, "desc": s.desc}
                             for s in plan.steps]},
            metadata={"plan_id": plan.plan_id, "steps_total": len(plan.steps)},
        ) as _kspan:
            try:
                await self._run_inner(plan)
            finally:
                try:
                    _kspan.update(metadata={
                        "state": plan.state, "note": plan.note,
                        "replans": plan.replans,
                        "steps_done": sum(1 for s in plan.steps if s.status == "done"),
                        "steps_failed": sum(1 for s in plan.steps if s.status == "failed"),
                        "verify_failures": plan.verify_failures,
                        "duration_s": round(time.time() - plan.created_ts, 1),
                    })
                except Exception:
                    pass

    async def _run_inner(self, plan: TaskPlan) -> None:
        from junjun_agent.task_kernel.planner import revise_remaining
        max_replans = int(_cfg().get("max_replans", 3))  # Phase 1：1 -> 3 配指数退避
        backoff_base = float(_cfg().get("replan_backoff_seconds", 5))
        try:
            while True:
                if time.time() > plan.deadline_ts:
                    plan.state = "failed"
                    plan.note = "超过时限"
                    break
                ready = plan.ready_steps()
                if not ready:
                    if any(s.status == "running" for s in plan.steps):
                        await asyncio.sleep(0.5)
                        continue
                    if any(s.status == "pending" for s in plan.steps):
                        plan.state = "failed"
                        plan.note = "步骤图无法推进（依赖断裂）"
                    break
                # 副作用/成品硬校验（Phase 1）：无交集步骤并行，副作用步骤
                # 串行殿后——双图事故的教训要代码兜底，不只靠 prompt 软约束。
                safe = [s for s in ready if s.action not in SIDE_EFFECT_ACTIONS]
                side = [s for s in ready if s.action in SIDE_EFFECT_ACTIONS]
                if safe:
                    await asyncio.gather(*(self._run_step(plan, s) for s in safe),
                                         return_exceptions=True)
                for s in side:
                    await self._run_step(plan, s)
                _persist(plan)
                failed = [s for s in plan.steps if s.status == "failed"]
                # 验证失败计数（指标进根 span）：区别于工具/网络类失败
                plan.verify_failures += sum(
                    1 for s in failed if s.error.startswith(("验证未通过", "验收不通过")))
                for s in failed:
                    if plan.attempts.get(s.id, 0) < 2:
                        logger.info(f"步骤 {s.id} 失败（{s.error[:60]}），重试一次")
                        s.status, s.error = "pending", ""
                    elif plan.replans < max_replans:
                        plan.replans += 1
                        logger.info(f"步骤 {s.id} 连续失败，局部重规划（第 {plan.replans} 次）")
                        if backoff_base > 0:  # 指数退避：瞬时故障给恢复窗口
                            await asyncio.sleep(
                                min(60.0, backoff_base * 2 ** (plan.replans - 1)))
                        new_steps = await revise_remaining(plan, s.desc, s.error)
                        if new_steps:
                            from junjun_agent.task_kernel.plan import merge_revisal
                            # 未显式放弃的原 pending 保留（防静默丢步骤/目标放弃）
                            merge_revisal(plan, new_steps)
                            _persist(plan)
                        else:
                            s.status = "failed"  # 重规划也废了，认输
                            plan.state = "failed"
                            plan.note = f"步骤「{s.desc[:30]}」失败且无法重规划"
                    else:
                        plan.state = "failed"
                        plan.note = f"步骤「{s.desc[:30]}」失败：{s.error[:100]}"
                if plan.state == "failed":
                    break
                if all(s.status in ("done", "skipped") for s in plan.steps):
                    plan.state = "done"
                    break
        except asyncio.CancelledError:
            plan.state, plan.note = "failed", "进程关停，任务取消"
            _persist(plan)
            raise
        except Exception as e:
            plan.state, plan.note = "failed", f"{type(e).__name__}: {e}"
            logger.warning(f"任务内核异常: {plan.note}")
        finally:
            await self._report(plan)
            self._plans.pop(plan.plan_id, None)
            _persist(plan)

    # ---------- 单步执行 ----------

    async def _run_step(self, plan: TaskPlan, step: Step) -> None:
        from junjun_core.observability import lf
        step.status = "running"
        plan.attempts[step.id] = plan.attempts.get(step.id, 0) + 1
        # 身份链（2026-08-13 审查 P1 双案实锤）：步骤执行不一定在发起消息的
        # context 里——审批恢复复制的是管理员的 context（is_admin_privileged
        # 全真、current_chat_id 是管理员私聊，后续步骤借管理员身份执行特权
        # 工具、结果流回群聊=跨会话隐私外泄）；断点恢复是空 context（workspace
        # 工具全落共享 unknown/ 目录且零报错）。步骤必须按 plan 发起者身份跑，
        # 人审只翻转批准位，不附带管理员身份。
        from junjun_core.security import set_caller
        from junjun_skills.builtin.memory_skills import current_chat_id
        # at_bot=True：任务只在被 @/私聊直呼（显式指令语义）时才路由进来，
        # 忠实还原提交时的权限判定（管理员在群里派单=当时的 @bot 指令）
        set_caller(plan.user_id, at_bot=True, is_group=plan.chat_id.endswith(":group"))
        current_chat_id.set(plan.chat_id)
        # 人审已批准的步骤置放行位（高危工具如 run_code 据此对非管理员放行）；
        # 本步骤执行完即复位，不泄漏到后续步骤。
        token = _kernel_step_approved.set(step.verify == "human" and step.approved)
        try:
            # 步骤级超时（Phase 2）：节点只在入口查计划死线，一个挂死的 LLM 调用
            # （限流重试风暴）能把图永远卡在 execute 里——计划死线够不着进行中的
            # await（2026-08-13 file-chart 评测 600s 僵尸实锤）。超时按普通失败
            # 处理，retry/replan 机器自然接管。
            timeout = float(_cfg().get("step_timeout_seconds", 300))
            await asyncio.wait_for(self._run_step_inner(plan, step), timeout=timeout)
        except TimeoutError:
            step.status = "failed"
            step.error = f"步骤执行超时（{timeout:.0f}s 无响应）"
            logger.info(f"步骤 {step.id} 超时按失败处理: {step.desc[:40]}")
        finally:
            _kernel_step_approved.reset(token)

    async def _run_step_inner(self, plan: TaskPlan, step: Step) -> None:
        from junjun_core.observability import lf
        with lf.start_span(
            name=f"task_kernel_step.{step.action}",
            input={"desc": step.desc, "args_hint": step.args_hint},
            metadata={"plan_id": plan.plan_id, "step_id": step.id,
                      "verify": step.verify, "attempt": plan.attempts[step.id]},
        ) as _sspan:
            try:
                if step.action == SYNTH_ACTION:
                    result = await self._synthesize(plan, step)
                else:
                    result = await self._call_tool(plan, step)
                ok = await self._verify(plan, step, result)
            except Exception as e:
                result, ok = "", False
                step.error = f"{type(e).__name__}: {e}"
            if ok:
                step.status = "done"
                # 大产出全文进材料库，上下文只留摘要+指针（PTC「中间数据
                # 不进城」移植）；短产出原样 inline。此前全文直接丢弃，
                # 合成/汇报只能拿 500 字残桩（materials.py 头注释）。
                from junjun_agent.task_kernel import materials
                step.result, step.material_id = materials.store_result(
                    plan.plan_id, step.id, result)
                logger.info(f"步骤 {step.id} 完成: {step.desc[:40]}")
            else:
                step.status = "failed"
                if not step.error:
                    step.error = "验证未通过" if result else "工具无产出"
                logger.info(f"步骤 {step.id} 失败: {step.error[:80]}")
            try:
                _sspan.update(metadata={"status": step.status, "error": step.error[:200]})
            except Exception:
                pass
            try:
                from junjun_core.observability import trajectory
                trajectory.emit("tk_step", plan.chat_id, plan_id=plan.plan_id,
                                step=step.id, action=step.action,
                                status=step.status, error=step.error[:100],
                                attempt=plan.attempts[step.id],
                                material=step.material_id)
            except Exception:
                pass

    async def _call_tool(self, plan: TaskPlan, step: Step) -> str:
        """按名找注册表工具直接调（继承注册处的超时/熔断/错误分类包装）。"""
        from junjun_skills.registry import get_tools
        tool = next((t for t in get_tools() if t.name == step.action), None)
        if tool is None:
            raise RuntimeError(f"工具 {step.action} 不在注册表（可能被熔断降级）")
        args = dict(step.args_hint)
        done = {s.id: s.result for s in plan.steps if s.status == "done"}
        # $步骤id 引用替换为前序产出
        for k, v in list(args.items()):
            if isinstance(v, str) and v.startswith("$"):
                ref = v[1:]
                if ref in done:
                    args[k] = done[ref]
        # 身份参数兜底（2026-08-15 eval 实锤）：规划器看不到下单上下文，
        # user_id 必填参数只能瞎编，DS 规划器干脆把 set_reminder 整步丢掉
        # （chain-weather-remind 连挂两轮）。非数字占位值（auto/对方/空）
        # 一律换成本单发起者；数字 QQ 保留——戳别人/改别人资料是合法的
        # 定向用法，不许误伤。
        if plan.user_id:
            fields = getattr(getattr(tool, "args_schema", None),
                             "model_fields", {}) or {}
            if "user_id" in fields:
                v = str(args.get("user_id", "") or "")
                if not v.isdigit():
                    if v:
                        logger.info(f"步骤 {step.id} user_id 占位值「{v[:20]}」"
                                    f"换成本单发起者")
                    args["user_id"] = plan.user_id
        if not args:
            args = self._default_args(tool, plan, step)
        out = await tool.ainvoke(args)
        text = out if isinstance(out, str) else str(out)
        # 注册表 _wrap_error_feedback 的现役格式是 "[TOOL_ERROR kind=...]"——
        # 只认 "[TOOL_ERROR]" 精确前缀会把错误文本当产出放行，步骤假成功、
        # 下游合成拿错误串当素材（2026-08-12 golden_tasks 评测实锤）。
        if text.startswith("[TOOL_ERROR"):
            raise RuntimeError(text[:200])
        return text

    @staticmethod
    def _default_args(tool, plan: TaskPlan, step: Step) -> dict:
        """规划器没给参数时，按工具 schema 的第一个字符串字段塞任务描述。"""
        try:
            fields = tool.args_schema.model_fields
            for name, f in fields.items():
                if f.annotation is str:
                    deps = "；".join(s.result[:100] for s in plan.steps
                                     if s.id in step.depends_on and s.result)
                    return {name: f"{step.desc}（任务目标：{plan.goal[:80]}）"
                                   + (f"；前序产出：{deps}" if deps else "")}
        except Exception:
            pass
        return {}

    async def _synthesize(self, plan: TaskPlan, step: Step) -> str:
        """llm_synthesize：纯文本合成步骤（调研报告、笔记汇总）。

        材料按需读回全文（材料库，单项/总量双预算）——此前只有 500 字
        残桩可写，报告内容饥荒；预算防超长材料烧穿合成上下文。
        """
        from junjun_llm import get_chat_model
        from langchain_core.messages import HumanMessage
        from junjun_agent.task_kernel import materials
        per_cap, total_cap = materials.synth_budget()
        chunks, used = [], 0
        for s in plan.steps:
            if s.id not in step.depends_on or not s.result:
                continue
            text = materials.material_text(s, per_cap)
            if used + len(text) > total_cap:
                text = text[:max(0, total_cap - used)]
            if not text:
                break
            chunks.append(f"【{s.desc}】\n{text}")
            used += len(text)
        deps = "\n".join(chunks)
        prompt = (f"任务目标：{plan.goal}\n\n前序步骤产出：\n{deps or '（无）'}\n\n"
                  f"当前步骤：{step.desc}\n\n直接产出这一步的成果内容。")
        model = get_chat_model("thinker")  # 步骤合成（报告/汇总）：开思考提质
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        return str(resp.content)

    async def _verify(self, plan: TaskPlan, step: Step, result: str) -> bool:
        if step.verify == "none":
            return True
        if not result.strip():
            return False
        if step.verify == "llm_judge":
            from junjun_llm import get_chat_model
            from langchain_core.messages import HumanMessage
            # 完成判据（Phase 1）：验收对准规划意图，避免泛泛的「内容太简略」
            # 误杀内容达标但措辞不合评委口味的产出。
            criteria = f"\n完成判据：{step.done_criteria}" if step.done_criteria else ""
            prompt = (f"判断下面的产出是否基本完成了步骤目标（只答「可以」或「不行+一句原因」）。\n"
                      f"步骤目标：{step.desc}{criteria}\n产出：\n{result[:1500]}")
            try:
                model = get_chat_model("utils_small") or get_chat_model("utils")
                resp = await model.ainvoke([HumanMessage(content=prompt)])
                verdict = str(resp.content)
                if verdict.strip().startswith("不行"):
                    step.error = f"验收不通过：{verdict[:100]}"
                    return False
            except Exception as e:
                # 验收调用本身炸了不当失败（工具结果已在）——但 2026-08-13 审查
                # P1：fail-open 不能无声，key 故障时全步骤「验收通过」汇报成功
                # 是教科书级掩盖。留痕：warning + plan 计数，终态汇报可见。
                plan.verify_skipped += 1
                logger.warning(f"步骤 {step.id} 验收调用异常按通过处理"
                               f"（本计划第 {plan.verify_skipped} 次未经验收）: "
                               f"{type(e).__name__}: {e}")
        return True

    # ---------- 汇报 ----------

    async def _report(self, plan: TaskPlan) -> None:
        """终态汇报：口吻过 persona_brief，内容照实（含失败原因与已完成部分）。"""
        from junjun_agent.outbound import send_proactive
        from junjun_agent.tasks import task_manager
        status = "done" if plan.state == "done" else "failed"
        done = [s for s in plan.steps if s.status == "done"]
        try:
            text = await self._compose_report(plan)
        except Exception as e:
            logger.warning(f"汇报合成异常，用模板兜底: {e}")
            text = self._fallback_report(plan)
        if plan.verify_skipped:
            # fail-open 留痕的落地一半（2026-08-13 审查 P1）：验收异常被按通过
            # 处理的步骤必须在终态汇报里点名——否则 key 故障时全步骤「验收通过」
            # 汇报成功，用户无从察觉。追加而非塞进合成提示：LLM 可能吞掉。
            text = (text or "") + \
                f"\n（{plan.verify_skipped} 步的验收环节没跑通，按产出直接算通过）"
        said = ""
        if text:
            # remember=True：终态汇报必须回填记忆+落库——2026-08-14 实锤：
            # 群里发出去的调研报告 bot 自己记不住，被追问「报告呢」只能装傻
            # （与 2026-08-04 P1-6 副作用失忆同类；async_jobs 汇报路径本就
            # remember=True，此处是一开始漏开的孤岛）。
            sent = await send_proactive(
                plan.chat_id, [ReplySegment(type="text", data=text)],
                source="task_kernel", remember=True)
            said = text if sent else ""
        if plan.state in ("done", "failed"):
            detail = (f"{len(done)}/{len(plan.steps)} 步完成"
                      + (f"，{plan.note}" if plan.note else ""))
            task_manager._record_outcome(plan.chat_id, "task_kernel", status,
                                         detail, said=said)
        logger.info(f"[{plan.chat_id}] 复杂任务终态 {plan.state}: {plan.goal[:40]} {plan.note}")
        try:
            from junjun_core.observability import trajectory
            trajectory.emit("tk_done", plan.chat_id, plan_id=plan.plan_id,
                            state=plan.state, note=plan.note[:100],
                            replans=plan.replans, verify_skipped=plan.verify_skipped,
                            steps_done=len(done), steps_total=len(plan.steps))
        except Exception:
            pass

    async def _compose_report(self, plan: TaskPlan) -> str:
        from junjun_llm import get_chat_model
        from junjun_agent.persona import persona_brief
        from junjun_core.config import get_global_config
        from langchain_core.messages import HumanMessage
        nickname = get_global_config().bot.nickname
        lines = plan.summary_lines()
        # 汇报要「给货」：有材料的步骤读材料全文（单项预算），不拿摘要充数
        from junjun_agent.task_kernel import materials
        per_cap = materials.report_budget()
        results = "\n".join(
            f"【{s.desc}】{materials.material_text(s, per_cap)[:per_cap]}"
            for s in plan.steps if s.status == "done" and s.result)
        if plan.state == "done":
            ask = f"任务已全部完成，把最终成果汇报给对方（成果内容为主，别只报喜不给货）。"
        else:
            ask = (f"任务没做完（{plan.note}）。照实说哪几步成了、卡在哪、"
                   f"已完成的部分如果有用也带给对方。")
        prompt = (f"你是「{nickname}」——{persona_brief()}\n\n"
                  f"你之前接了一个任务：{plan.goal}\n\n步骤情况：\n" + "\n".join(lines)
                  + f"\n\n产出：\n{results or '（无）'}\n\n{ask}\n"
                    f"用你平时的口气说，别太长。")
        model = get_chat_model("utils")
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        return str(resp.content).strip()

    @staticmethod
    def _fallback_report(plan: TaskPlan) -> str:
        lines = plan.summary_lines()
        if plan.state == "done":
            return "弄好了：\n" + "\n".join(lines)
        return f"这个任务没弄完（{plan.note or '未知原因'}）：\n" + "\n".join(lines)


kernel = TaskKernel()
