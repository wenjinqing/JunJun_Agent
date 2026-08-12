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
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_PLANNER_PROMPT = """你是任务规划器。把用户的委托拆成可执行的步骤图，输出纯 JSON（不要任何其他文字）。

用户委托：{goal}

可用工具（action 只能从这里选，或用 "llm_synthesize" 表示直接文本合成；括号里是参数签名，? 结尾=可选）：
{tool_list}

规则：
1. 最多 {max_steps} 步，能少不多；每步一句话说清要做什么（desc）。
2. 步骤间有先后依赖就写 depends_on（只允许依赖前面的步骤）；无依赖的步骤会并行。
3. 画图/语音这类成品工具（ai_draw、unified_tts 等）只能放最后一步——它们是即交即走的。发说说/发空间类需求直接一步 send_feed（它内部自带配图能力），不要拆 ai_draw→send_feed 两步（会画两张图）。
4. verify 填验证方式：tool_ok（工具不报错即可，默认）/ llm_judge（需要判断产出质量，如报告、笔记）/ none / human（发空间、订阅推送这类对外发布动作——必须等管理员批准才执行）。
5. args_hint 必须严格按工具签名给参数：参数名照抄签名（必填的一个都不许漏），值给具体内容或「$步骤id」引用前序产出。拿不准可选参数就省略。
6. 每步写 done_criteria：这一步凭什么算完成（一句话、可检查），验收时用它对准你的意图。

输出格式：
{{"steps": [{{"id": "s1", "action": "工具名", "desc": "做什么", "args_hint": {{}}, "depends_on": [], "verify": "tool_ok", "done_criteria": "凭什么算完成"}}]}}"""


def _extract_json(text: str) -> dict:
    """规划器产出清洗：对象/数组两种形态都认，按起点谁早取谁。

    2026-08-12 实锤：revise 提示只写「格式同前」没给格式，GLM 自由发挥产出
    裸数组——对象正则会抓到数组【元素】当计划（没有 steps 键照旧全废），
    所以对象缺 steps 时继续看数组兜底（parse_plan 仍清洗非法步骤，宽进严出）。
    """
    text = text or ""
    candidates = []
    for rx, kind in ((_JSON_RE, "obj"), (_ARRAY_RE, "arr")):
        m = rx.search(text)
        if m:
            candidates.append((m.start(), kind, m.group(0)))
    for _, kind, blob in sorted(candidates):
        try:
            payload = json.loads(blob)
        except Exception:
            continue
        if kind == "obj":
            if isinstance(payload, dict) and "steps" in payload:
                return payload
            continue  # 缺 steps 的对象：可能是裸数组的元素，让数组兜底
        if isinstance(payload, list):
            return {"steps": payload}
    return {}


def _schema_brief(t) -> str:
    """工具参数签名摘要（Phase 1）：名字+类型+可问号，给规划器抄参数名的依据。

    2026-08-12 基线实锤：只给名字+一句描述时规划器瞎猜参数名（bilibili_summary
    缺 url / deep_research 缺 topic / send_feed 缺 content），3/10 失败全死于
    args_schema 校验。"""
    fields = getattr(getattr(t, "args_schema", None), "model_fields", None)
    if not fields:
        return ""
    parts = []
    for n, f in fields.items():
        ann = getattr(f.annotation, "__name__", None) or str(f.annotation)
        try:
            req = f.is_required()
        except Exception:
            req = True
        parts.append(f"{n}:{ann}" + ("" if req else "?"))
    return "(" + ", ".join(parts) + ")"


def _tool_catalog() -> str:
    """给规划器的工具清单（名字 + 参数签名摘要 + 一句描述）；熔断降级的工具不出列。"""
    from junjun_skills.registry import get_tools
    lines = []
    for t in get_tools():
        desc = (t.description or "").strip().split("\n")[0][:60]
        lines.append(f"- {t.name}{_schema_brief(t)}: {desc}")
    return "\n".join(lines)


_PLANNER_MAX_TOKENS = 8192
# 思考链会烧槽位 4096 上限把 JSON 截死（2026-08-12 revise 两次 output=4096 整
# 实锤）——规划/重规划调用单独放宽；上限只是允许更多，不强制烧满。


def _bound(model):
    """按调用放宽 max_tokens；测试假模型没有 bind，原样返回。"""
    return model.bind(max_tokens=_PLANNER_MAX_TOKENS) if hasattr(model, "bind") else model


async def make_plan(goal: str, *, chat_id: str, user_id: str,
                    max_steps: int = 6, model=None, callbacks=None) -> TaskPlan | None:
    """生成步骤图。失败/非法返回 None（调用方回退对话通道）。

    产出不合法时带提醒重试一次：规划是低频高价值调用，偶发的散文包裹/编造
    工具名值得一次纠偏机会（2026-08-12 基线 2/10 失败死于单次 None）。"""
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("thinker")  # 规划是低频高价值：开思考的 ***REMOVED***
    from langchain_core.messages import HumanMessage
    from junjun_skills.registry import get_tools
    valid = {t.name for t in get_tools()}
    prompt = _PLANNER_PROMPT.format(goal=goal, tool_list=_tool_catalog(),
                                    max_steps=max_steps)
    for attempt in (1, 2):
        resp = await _bound(model).ainvoke([HumanMessage(content=prompt)],
                                           config={"callbacks": callbacks or []})
        payload = _extract_json(str(resp.content))
        if payload:
            plan = parse_plan(payload, goal=goal, chat_id=chat_id, user_id=user_id,
                              valid_actions=valid, max_steps=max_steps)
            if plan is not None:
                return plan
        if attempt == 1:
            logger.info(f"规划器首次未产出合法计划，追加提醒重试: {goal[:40]}")
            prompt += ("\n\n提醒：上一次输出无法解析成合法计划。只输出纯 JSON，不要"
                       "任何解释文字；action 必须从可用工具清单照抄，必填参数一个不漏。")
    logger.warning(f"规划器两次均未产出合法计划，回退对话通道: {goal[:40]}")
    return None


_REVISER_PROMPT = """你是任务规划器。一个执行中的计划有步骤失败了，给出修正后的【剩余】步骤（纯 JSON，不要任何其他文字）。

任务目标：{goal}
失败的步骤：{failed_desc}
失败原因：{error}
已完成的步骤及产出：
{done_digest}
原计划中未执行的步骤：
{pending_digest}

要求：
1. id 从 r1 开始重新编号；depends_on 只允许指向已完成步骤 id 或更新的新步骤 id。
2. action 从可用工具里选，或用 "llm_synthesize"；args_hint 严格按工具签名给参数。
3. 若能直接利用已完成产出收尾，可以只给一步 llm_synthesize。

输出格式（照这个写，别自造字段名）：
{{"steps": [{{"id": "r1", "action": "工具名", "desc": "做什么", "args_hint": {{}}, "depends_on": [], "verify": "tool_ok", "done_criteria": "凭什么算完成"}}]}}"""


async def revise_remaining(plan: TaskPlan, failed_step_desc: str, error: str,
                           *, model=None, callbacks=None) -> list | None:
    """局部重规划：只重写未执行的剩余步骤。返回新的 Step 列表或 None。"""
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("thinker")  # 局部重规划同理：开思考
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
    resp = await _bound(model).ainvoke([HumanMessage(content=prompt)],
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
