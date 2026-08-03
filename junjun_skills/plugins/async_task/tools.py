"""async_task 插件：异步任务队列的 LLM 入口 + 「agent 型 job」首个实现。

- run_background_task：主 Agent 接单（落 AsyncJob 表，后台执行，完成主动汇报）
- agent_task handler：隔离上下文的一次性子 agent——没有人格、不碰用户、
  只有只读工具（搜索/时间）、独立迭代预算，跑完即焚。
  用户听到的永远只有君君一张嘴（完成时由君君口吻汇报）。
- /tasks /canceltask：确定性查询/取消通道（与 /subs /unsub 同构）
- sweep：60s 调度兜底（重启恢复、崩溃残留回炉、尸体清理）
"""

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_agent.loop import async_jobs
from junjun_agent.loop.scheduler import ScheduledTask, scheduler
from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.async_task")

_SWEEP_INTERVAL = 60  # 秒


# ---------------------------------------------------------------- agent 型 job
# 子 agent 只给只读信息工具：它没有人格也没有会话上下文，
# 发消息/记忆/空间这类带副作用的工具一律不给（汇报由队列引擎统一做）。

def _safe_tools() -> list:
    from junjun_skills.registry import get_tools
    out = []
    for t in get_tools():
        if t.name in ("web_search", "get_time"):
            out.append(t)
        elif t.name.startswith("mcp_") and "search" in t.name:
            out.append(t)
    return out


_SUBAGENT_PROMPT = """你是后台任务执行助手（没有人格，输出不会直接发给人看，会有人帮你转述）。
完成这个任务并输出结果正文：
任务：{task}
当前时间：{now}

要求：
- 需要资料就用搜索工具，多查几个来源交叉验证，别凭记忆编
- 直接输出结果正文（中文，条理清晰，不超过 800 字）
- 不要解释你做了什么过程，不要说「作为 AI」"""


async def _agent_task_handler(job, payload: dict, *, model=None) -> str:
    """隔离子 agent 跑自然语言任务，返回结果正文（抛异常=失败，由引擎兜底）。"""
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("agent")
    from datetime import datetime
    agent = create_agent(model=model, tools=_safe_tools())
    cfg = get_global_config().raw.get("async_task", {}) or {}
    max_iter = int(cfg.get("subagent_max_iter", 12))
    invoke_cfg = {"recursion_limit": 2 * max_iter + 1}
    try:
        from junjun_llm import get_callbacks
        invoke_cfg["callbacks"] = get_callbacks()
        invoke_cfg["metadata"] = {"chat_id": job.chat_id,
                                  "langfuse_session_id": job.chat_id,
                                  "langfuse_tags": ["junjun", "subagent", job.kind]}
    except Exception:
        pass
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=_SUBAGENT_PROMPT.format(
            task=str(payload.get("task") or job.title)[:500],
            now=datetime.now().strftime("%Y-%m-%d %H:%M")))]},
        config=invoke_cfg)
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and not (m.tool_calls or []):
            text = m.content or ""
            if isinstance(text, list):
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            text = str(text).strip()
            if text:
                return text
    raise RuntimeError("子任务没有产出结果")


async_jobs.register_handler("agent_task", _agent_task_handler)

from junjun_skills.plugins.async_task import research  # noqa: E402

async_jobs.register_handler("deep_research", research.deep_research_handler)


# ---------------------------------------------------------------- LLM 工具

@tool
def run_background_task(task: str) -> str:
    """把耗时任务丢到后台执行，完成后你会主动发消息向对方汇报结果。这是「派活」的通用入口：
    读长文、盯梢类跟进等一时半会儿做不完的活用本工具。
    用户说「帮我深入查查/慢慢做不着急/做好了叫我」时也必须调用本工具接单。
    注意：口头答应不等于接了活——不调本工具就没有任何任务在执行，绝不许假装已经在做。
    调研/出报告类请求用 deep_research（多轮检索+阅读原文，报告质量更高）；
    认真看 B 站/抖音视频用 watch_video；一两分钟能搞定的小事（查天气、搜条快讯）直接自己做。

    Args:
        task: 任务的完整描述（要做什么、做到什么程度），后台执行时看不到聊天记录，要写全
    """
    from junjun_skills.builtin.memory_skills import current_chat_id
    from junjun_core.security import current_user_id, current_nickname
    task = (task or "").strip()
    if len(task) < 4:
        return "任务描述太短了，后台执行时看不到聊天记录，把「要做什么」写完整再派。"
    job, err = async_jobs.submit_job(
        "agent_task", title=task[:80], payload={"task": task},
        chat_id=current_chat_id.get(),
        user_id=current_user_id.get(), nickname=current_nickname.get())
    if job is None:
        return err
    return (f"接单成功：任务 #{job.job_id}（{job.title}）已在后台开始执行。"
            f"完成后你会主动在这个会话汇报结果。现在请告诉对方：已接单，做好了会来叫 ta。")


@tool
def deep_research(topic: str) -> str:
    """深度研究：拆查询 -> 多源检索 -> 阅读原文 -> 交叉验证 -> 写带来源链接的研究报告，
    后台执行完成后主动汇报。用户说「深度研究/调研/整理一份报告/系统查查」时用本工具。
    这是「出报告」唯一真正的入口：口头答应不等于接了活。
    一两分钟能搞定的小事（查天气、搜条快讯）直接自己做，不要用本工具；
    读长文/长视频等其他耗时活用 run_background_task。

    Args:
        topic: 研究主题（要研究什么、关心哪些方面），后台执行时看不到聊天记录，要写具体
    """
    from junjun_skills.builtin.memory_skills import current_chat_id
    from junjun_core.security import current_user_id, current_nickname
    topic = (topic or "").strip()
    if len(topic) < 4:
        return "研究主题太短了，后台执行时看不到聊天记录，把「要研究什么」写具体再派。"
    job, err = async_jobs.submit_job(
        "deep_research", title=topic[:80], payload={"topic": topic},
        chat_id=current_chat_id.get(),
        user_id=current_user_id.get(), nickname=current_nickname.get())
    if job is None:
        return err
    return (f"接单成功：深度研究 #{job.job_id}（{job.title}）已在后台开始执行，"
            f"完成后你会主动在这个会话汇报研究报告。现在请告诉对方：已接单，做好了会来叫 ta。")


@tool
def list_background_tasks() -> str:
    """查看当前会话的后台任务（排队中/执行中/最近完成）。用户问「任务做得怎么样了/做好了吗」时使用。"""
    from junjun_skills.builtin.memory_skills import current_chat_id
    return async_jobs.list_for_chat(current_chat_id.get())


@tool
def cancel_background_task(job_id: str) -> str:
    """取消后台任务。用户说「那个任务别做了/取消任务」时使用。

    Args:
        job_id: 任务编号（list_background_tasks 可查）
    """
    from junjun_core.security import current_user_id
    return async_jobs.cancel_job(job_id, current_user_id.get())


TOOLS = [deep_research, run_background_task, list_background_tasks, cancel_background_task]


# ---------------------------------------------------------------- 命令

@register_command("tasks", aliases=["任务"], plugin="async_task",
                  description="查看本会话后台任务")
async def tasks_cmd(ctx) -> str:
    return async_jobs.list_for_chat(ctx.session.chat_id)


@register_command("canceltask", aliases=["取消任务"], plugin="async_task",
                  description="取消后台任务（/canceltask <编号>）")
async def canceltask_cmd(ctx) -> str:
    from junjun_core.security import current_user_id
    if not (ctx.args or "").strip():
        return "用法：/canceltask <编号>（/tasks 查编号）"
    return async_jobs.cancel_job(ctx.args.strip(), current_user_id.get())


scheduler.add(ScheduledTask("asyncjob_sweep", async_jobs.sweep_jobs,
                            interval=_SWEEP_INTERVAL, plugin="async_task"))
