"""TaskKernel 规划器：LLM 产出结构化步骤图（utils 槽，无人设——能力与人格分层）。

规划/验证是能力活，不带君君的口吻；口吻只出现在接单话术与最终汇报
（executor 里过 persona_brief）。
"""

import json
import re

from junjun_core.observability import get_logger
from junjun_agent.task_kernel.plan import (
    ASYNC_JOB_TAG, SYNTH_ACTION, TaskPlan, parse_plan,
)

logger = get_logger("task_kernel.planner")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_PLANNER_PROMPT = """你是任务规划器。把用户的委托拆成可执行的步骤图，输出纯 JSON（不要任何其他文字）。

用户委托：{goal}
下单人 QQ：{user_id}（工具签名要 user_id 参数时填它——除非委托里明确指了别人；不许因为填不出就丢掉整步）

可用工具（action 只能从这里选，或用 "llm_synthesize" 表示直接文本合成；括号里是参数签名，? 结尾=可选）：
{tool_list}

规则：
1. 最多 {max_steps} 步，能少不多；每步一句话说清要做什么（desc）。
   但「能少不多」不许吃掉委托里的动作意图：每个要落地的动作（查/设提醒/发布/写）各成一步；
   条件句里的后续动作（「如果下雨就提醒我」的提醒）也必须独立成步骤——条件写进 desc/参数里
   （如提醒内容带「如果明天下雨」），不许因为「要看上一步结果」就吞掉整步
   （2026-08-15 eval 实锤：压缩成纯查天气，提醒意图丢了）。
2. 步骤间有先后依赖就写 depends_on（只允许依赖前面的步骤）；无依赖的步骤会并行。
3. 画图/语音这类成品工具（ai_draw、unified_tts 等）只能放最后一步——它们是即交即走的。发说说/发空间类需求直接一步 send_feed（它内部自带配图能力），不要拆 ai_draw→send_feed 两步（会画两张图）。
   标［异步接单］的工具（deep_research、run_background_task、watch_video）同样只能放最后一步：它返回的只是接单回执，真正的成果由后台任务自己做、做完自己直接发给对方——它之后不要再排任何步骤，尤其不要排 llm_synthesize 去「写报告/汇总」（没有材料只能编空话）；调研/出报告类需求整体就是一步 deep_research。
   有专职工具覆盖的需求必须用专职工具（画图 ai_draw、看视频 watch_video、深研 deep_research）；run_background_task 只兜底没有专职工具覆盖的泛化长任务——拿它代替 ai_draw 画图等于绕开成品工具的验收链（2026-08-15 eval chain-draw-report 实锤）。
4. verify 填验证方式：tool_ok（工具不报错即可，默认）/ llm_judge（步骤产出本身就是可判质量的文本，如报告、笔记、感想）/ none / human（发空间、订阅推送这类对外发布动作——必须等管理员批准才执行）。
   注意：工具返回的多是确认语或原始素材，不含个人创作——「写得好不好、有没有感受」这类判断对象应该是 llm_synthesize 步骤；给工具步骤配 llm_judge 等于逼它返回它生产不了的东西，必死。
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
    """给规划器的工具清单（名字 + 参数签名摘要 + 一句描述）；熔断降级的工具不出列。

    异步接单工具的前置标记必须在截断之前——desc 只取首行 60 字，而
    deep_research 的「后台执行完成后主动汇报」写在 docstring 第二行，
    2026-08-14 生产实锤：规划器看不到异步属性，把它当同步材料源排进链路。
    """
    from junjun_skills.registry import get_tools
    lines = []
    for t in get_tools():
        desc = (t.description or "").strip().split("\n")[0][:60]
        mark = ("［异步接单：只回接单回执，成果由后台自己做并直接汇报］"
                if ASYNC_JOB_TAG in (getattr(t, "tags", None) or []) else "")
        lines.append(f"- {t.name}{_schema_brief(t)}: {mark}{desc}")
    return "\n".join(lines)


def _async_actions() -> frozenset:
    """当前注册的异步接单工具名集合（tool.tags 带 ASYNC_JOB_TAG）。"""
    from junjun_skills.registry import get_tools
    return frozenset(t.name for t in get_tools()
                     if ASYNC_JOB_TAG in (getattr(t, "tags", None) or []))


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
                                    max_steps=max_steps,
                                    user_id=user_id or "未知（评测/缺上下文，填 auto 占位）")
    for attempt in (1, 2):
        resp = await _bound(model).ainvoke([HumanMessage(content=prompt)],
                                           config={"callbacks": callbacks or []})
        payload = _extract_json(str(resp.content))
        if payload:
            plan = parse_plan(payload, goal=goal, chat_id=chat_id, user_id=user_id,
                              valid_actions=valid, max_steps=max_steps,
                              async_actions=_async_actions())
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
下单人 QQ：{user_id}（要 user_id 参数的步骤填它，除非委托明确指了别人）
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
4. 【剩余步骤全都要列】：原计划里未执行但仍然要做的步骤，用原 id 原依赖照列上来——
   你只列了修正步骤的话，没列的原步骤会被当「确认放弃」处理。确认不再需要的步骤，
   显式写进 "drop": ["原步骤id"]。
5. 验收失败（验收不通过）的修法是换验证方式或换成 llm_synthesize 重写，不是换参数重试。
6. ［异步接单］工具（deep_research、run_background_task、watch_video）只能放最后一步：
   它返回的是接单回执不是材料，成果由后台自己做并直接汇报——别在它后面排汇总/写报告步骤。

输出格式（照这个写，别自造字段名）：
{{"steps": [{{"id": "r1", "action": "工具名", "desc": "做什么", "args_hint": {{}}, "depends_on": [], "verify": "tool_ok", "done_criteria": "凭什么算完成"}}], "drop": []}}"""


class Revisal:
    """局部重规划产出：新步骤 + 显式放弃的步骤 id。

    drop 是模型【显式声明】的放弃清单；没在 drop 里也没被重列的原 pending
    步骤由调用方保留——不声明就丢步骤 = 目标静默放弃（2026-08-12 实锤：
    send_feed 人审步骤被重规划悄悄吞掉，任务「完成」了但说说没发）。
    """

    def __init__(self, steps: list, drop: list):
        self.steps = list(steps)
        self.drop = [str(d) for d in drop]


async def revise_remaining(plan: TaskPlan, failed_step_desc: str, error: str,
                           *, model=None, callbacks=None) -> Revisal | None:
    """局部重规划：只重写未执行的剩余步骤。返回 Revisal 或 None。"""
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
        user_id=plan.user_id or "未知（填 auto 占位）",
        done_digest=done_digest, pending_digest=pending_digest)
    resp = await _bound(model).ainvoke([HumanMessage(content=prompt)],
                                       config={"callbacks": callbacks or []})
    payload = _extract_json(str(resp.content))
    if not payload:
        return None
    drop = payload.get("drop") if isinstance(payload.get("drop"), list) else []
    from junjun_skills.registry import get_tools
    valid = {t.name for t in get_tools()}
    done_ids = {s.id for s in plan.steps if s.status == "done"}
    revised = parse_plan(payload, goal=plan.goal, chat_id=plan.chat_id,
                         user_id=plan.user_id, valid_actions=valid,
                         async_actions=_async_actions())
    if revised is None:
        return None
    # 依赖修正：parse_plan 只认新计划内部的前向 id，指向已完成步骤的依赖在
    # 入口就被剥掉了——从原始 payload 收回 reviser 的真实声明，已完成/新步骤
    # 内的才保留（此前 done_ids 分支是死代码，重规划的合成步骤一直拿不到材料，
    # 2026-08-15 eval research-video-notes 实锤）。
    raw_deps = {str(rs.get("id") or ""): [str(d) for d in (rs.get("depends_on") or [])]
                for rs in payload.get("steps", []) if isinstance(rs, dict)}
    new_ids = {s.id for s in revised.steps}
    for s in revised.steps:
        s.depends_on = [d for d in raw_deps.get(s.id, s.depends_on)
                        if d in done_ids or d in new_ids]
    # 合成步骤材料兜底（2026-08-15 eval research-video-notes 实锤）：重规划
    # 产出的 llm_synthesize 若依赖被上面剥光（模型指向了失败/不存在的步骤），
    # 就在零材料下合成——要么编造要么摆烂「我没收到材料」，验收还容易漏过。
    # 合成步骤的材料只能来自已完成产出，剥光即断供：自动挂到所有带产出的
    # 已完成步骤。只兜合成步骤——工具步骤无依赖是正常形态，不许挂。
    done_with_result = [s.id for s in plan.steps
                        if s.status == "done" and s.result]
    if done_with_result:
        for s in revised.steps:
            if s.action == SYNTH_ACTION and not s.depends_on:
                s.depends_on = list(done_with_result)
                logger.info(f"重规划合成步骤 {s.id} 依赖被剥光，"
                            f"自动挂接已完成产出 {done_with_result}")
    return Revisal(revised.steps, drop)
