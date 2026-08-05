"""TaskKernel 规划器：LLM 产出结构化步骤图（utils 槽，无人设——能力与人格分层）。

规划/验证是能力活，不带君君的口吻；口吻只出现在接单话术与最终汇报
（executor 里过 persona_brief）。
"""

import json
import re

from junjun_core.observability import get_logger
from junjun_agent.task_kernel.plan import TaskPlan, parse_plan

logger = get_logger("task_kernel.planner")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_PLANNER_PROMPT = """你是任务规划器。把用户的委托拆成可执行的步骤图，输出纯 JSON（不要任何其他文字）。

用户委托：{goal}

可用工具（action 只能从这里选，或用 "llm_synthesize" 表示直接文本合成）：
{tool_list}

规则：
1. 最多 {max_steps} 步，能少不多；每步一句话说清要做什么（desc）。
2. 步骤间有先后依赖就写 depends_on（只允许依赖前面的步骤）；无依赖的步骤会并行。
3. 画图/语音这类成品工具（ai_draw、unified_tts 等）本身是最后一步；但发布/发送类副作用工具（send_feed 等）可以紧随其后。
4. verify 填验证方式：tool_ok（工具不报错即可，默认）/ llm_judge（需要判断产出质量，如报告、笔记）/ none。
5. args_hint 给工具的参数提示（如搜索关键词），可以给后序步骤引用前序结果写「$步骤id」。

输出格式：
{{"steps": [{{"id": "s1", "action": "工具名", "desc": "做什么", "args_hint": {{}}, "depends_on": [], "verify": "tool_ok"}}]}}"""


def _extract_json(text: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _tool_catalog() -> str:
    """给规划器的工具清单（名字 + 一句描述）；熔断降级的工具不出列。"""
    from junjun_skills.registry import get_tools
    lines = []
    for t in get_tools():
        desc = (t.description or "").strip().split("\n")[0][:60]
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


async def make_plan(goal: str, *, chat_id: str, user_id: str,
                    max_steps: int = 6, model=None, callbacks=None) -> TaskPlan | None:
    """生成步骤图。失败/非法返回 None（调用方回退对话通道）。"""
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
    from langchain_core.messages import HumanMessage
    prompt = _PLANNER_PROMPT.format(goal=goal, tool_list=_tool_catalog(),
                                    max_steps=max_steps)
    resp = await model.ainvoke([HumanMessage(content=prompt)],
                               config={"callbacks": callbacks or []})
    payload = _extract_json(str(resp.content))
    if not payload:
        logger.warning(f"规划器未产出 JSON，回退对话通道: {goal[:40]}")
        return None
    from junjun_skills.registry import get_tools
    valid = {t.name for t in get_tools()}
    plan = parse_plan(payload, goal=goal, chat_id=chat_id, user_id=user_id,
                      valid_actions=valid, max_steps=max_steps)
    if plan is None:
        logger.warning(f"规划器产出无合法步骤，回退对话通道: {goal[:40]}")
    return plan


_REVISER_PROMPT = """你是任务规划器。一个执行中的计划有步骤失败了，给出修正后的【剩余】步骤（纯 JSON）。

任务目标：{goal}
失败的步骤：{failed_desc}
失败原因：{error}
已完成的步骤及产出：
{done_digest}
原计划中未执行的步骤：
{pending_digest}

输出修正后的剩余步骤图（格式同前，id 从 r1 开始重新编号，依赖只允许指向已完成步骤 id 或新步骤 id）。
若能直接利用已完成产出收尾，可以只给一步 llm_synthesize。"""


async def revise_remaining(plan: TaskPlan, failed_step_desc: str, error: str,
                           *, model=None, callbacks=None) -> list | None:
    """局部重规划：只重写未执行的剩余步骤。返回新的 Step 列表或 None。"""
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
    from langchain_core.messages import HumanMessage
    done_digest = "\n".join(
        f"- [{s.id}] {s.desc}: {s.result[:100]}" for s in plan.steps if s.status == "done"
    ) or "（无）"
    pending_digest = "\n".join(
        f"- [{s.id}] {s.desc}" for s in plan.steps if s.status == "pending"
    ) or "（无）"
    prompt = _REVISER_PROMPT.format(
        goal=plan.goal, failed_desc=failed_step_desc, error=error[:200],
        done_digest=done_digest, pending_digest=pending_digest)
    resp = await model.ainvoke([HumanMessage(content=prompt)],
                               config={"callbacks": callbacks or []})
    payload = _extract_json(str(resp.content))
    if not payload:
        return None
    from junjun_skills.registry import get_tools
    valid = {t.name for t in get_tools()}
    done_ids = {s.id for s in plan.steps if s.status == "done"}
    revised = parse_plan(payload, goal=plan.goal, chat_id=plan.chat_id,
                         user_id=plan.user_id, valid_actions=valid)
    if revised is None:
        return None
    # 依赖修正：引用已完成步骤的依赖保留，其余依赖只认新步骤内部
    new_ids = {s.id for s in revised.steps}
    for s in revised.steps:
        s.depends_on = [d for d in s.depends_on if d in done_ids or d in new_ids]
    return revised.steps
