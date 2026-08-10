"""daily_report 插件：每天一条「热点日报」说说——选题→深研→写稿→人审→发空间。

与 junzone_auto_post 的分工：
- auto_post：日记体/主题随想，轻量直接发（像真人的碎碎念）
- daily_report：抓真实热点（news 60s + RSS）→ deep_research 同款深研 →
  君君口吻成稿 → 管理员人审 → 发空间。重内容、对外可见，所以必须人审。

管线本体在 graph.py（LangGraph：崩溃断点续跑 + 人审中断），本文件提供
节点依赖的默认实现（素材/选题/写稿/发布）与调度入口。

状态文件（DATA_DIR 下，测试可 monkeypatch）：
- last_run.json   {"date": ...} 当天已触发标记（防同一分钟重复触发）
- history.json    已发布日报 [{date, topic, tid}]（选题去重 + 防同日重发）

配置 [daily_report]：enable / time="21:30" / approval_timeout_seconds=600
/ max_titles=15 / history_days=7
"""

import json
import time
from pathlib import Path

from junjun_agent.commands import register_command
from junjun_agent.loop.scheduler import ScheduledTask, scheduler
from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.daily_report")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "daily_report"

_HISTORY_KEEP = 30   # 历史保留条数（选题去重只看 history_days 天内的）


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("daily_report", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 状态读写

def _read_json(name: str, default):
    path = DATA_DIR / name
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"日报状态读取失败 {name}: {e}")
    return default


def _write_json(name: str, data) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    except Exception as e:
        logger.warning(f"日报状态落盘失败 {name}: {e}")


def record_history(date: str, topic: str, tid: str) -> None:
    """发布成功落历史（graph._after_run 调用）。"""
    hist = _read_json("history.json", [])
    if not isinstance(hist, list):
        hist = []
    hist.append({"date": date, "topic": topic, "tid": tid, "ts": time.time()})
    _write_json("history.json", hist[-_HISTORY_KEEP:])


def recent_topics() -> list:
    """近 N 天已发日报的主题（选题去重）。"""
    days = int(_cfg().get("history_days", 7))
    cutoff = time.time() - days * 86400
    hist = _read_json("history.json", [])
    if not isinstance(hist, list):
        return []
    return [str(h.get("topic") or "") for h in hist
            if float(h.get("ts", 0) or 0) >= cutoff and h.get("topic")]


# ---------------------------------------------------------------- 节点默认实现

async def gather_materials() -> list:
    """热点素材：news 60s 标题 + topic_finder RSS 标题，合并去重截断。"""
    titles: list[str] = []
    try:
        from junjun_skills.plugins.news.tools import fetch_60s_news
        news = await fetch_60s_news()
        if news:
            titles += [str(n).strip() for n in news.get("news") or [] if str(n).strip()]
    except Exception as e:
        logger.warning(f"日报素材：60s 新闻拉取失败: {type(e).__name__}: {e}")
    try:
        from junjun_skills.plugins.topic_finder.tools import fetch_rss
        tf_cfg = get_global_config().raw.get("topic_finder", {}) or {}
        titles += await fetch_rss(tf_cfg)
    except Exception as e:
        logger.warning(f"日报素材：RSS 拉取失败: {type(e).__name__}: {e}")
    seen, out = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:int(_cfg().get("max_titles", 15))]


_PICK_PROMPT = """你是热点编辑。从下面的今日热点标题里挑 1 个最值得「深研后聊给大家听」的话题。
标准：有信息增量（不是纯情绪/明星八卦）、大家会关心、适合展开聊 2-3 个知识点。
{recent_part}
标题列表：
{titles}

只输出 JSON：{{"topic": "话题（≤20字）"}} 或 {{"topic": ""}}（都不值得聊）。不要输出别的。"""


def _parse_topic(raw: str) -> str:
    import re
    m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not m:
        return ""
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return ""
    return str(data.get("topic") or "").strip()[:30]


async def pick_topic(titles: list, recent: list, *, model=None) -> str:
    """thinker 选题；失败/空 = 今天不出日报（返回 ""）。"""
    if not titles:
        return ""
    try:
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("thinker")
        from langchain_core.messages import HumanMessage
        recent_part = (f"最近几天已经聊过这些，不许重复：{'、'.join(recent)}\n"
                       if recent else "")
        resp = await model.ainvoke([HumanMessage(content=_PICK_PROMPT.format(
            recent_part=recent_part,
            titles="\n".join(f"- {t}" for t in titles)))])
        return _parse_topic(str(resp.content))
    except Exception as e:
        logger.warning(f"日报选题失败（今天跳过）: {type(e).__name__}: {e}")
        return ""


_WRITE_PROMPT = """{personality}

你今天刷到一个热点并认真查了资料，现在要在自己的 QQ 空间发一条「热点日报」说说，
把这个话题聊给好友听。{style}

话题：{topic}
你查到的材料（只许用这里面的信息，拿不准的别写）：
{report}

要求：
- 你的口吻：先一两句抛出话题（像跟朋友说「今天刷到个事」），再聊你挖到的关键
  信息和你自己的看法，口语化，不要播音腔/标题党/「大家好」
- 正文 ≤250 字
- 不要来源列表、不要链接（说说不是论文）
- 只输出说说正文，不要任何前后缀、引号、括号注释"""


async def write_draft(topic: str, report: str, *, model=None) -> str:
    """thinker 按人设成稿；返回 "" = 失败。"""
    try:
        from junjun_skills.plugins.junzone.tools import _persona
        personality, style = _persona()
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("thinker")
        from langchain_core.messages import HumanMessage
        resp = await model.ainvoke([HumanMessage(content=_WRITE_PROMPT.format(
            personality=personality, style=style, topic=topic,
            report=str(report)[:6000]))])
        return str(resp.content).strip()
    except Exception as e:
        logger.warning(f"日报写稿失败: {type(e).__name__}: {e}")
        return ""


async def publish_draft(draft: str):
    """发空间（复用 junzone 登录态与重试）。返回 tid 或 None。"""
    from junjun_skills.plugins.junzone.tools import _with_auth_retry, publish_feed, _bot_uin
    return await _with_auth_retry(publish_feed, _bot_uin(), draft)


# ---------------------------------------------------------------- 调度入口

async def daily_report_tick() -> None:
    """每分钟检查：到点且今天没跑过 -> 起一单日报（图管线，含人审）。"""
    cfg = _cfg()
    if not bool(cfg.get("enable", False)):
        return
    from datetime import datetime
    now = datetime.now()
    if now.strftime("%H:%M") != str(cfg.get("time", "21:30")):
        return
    today = now.strftime("%Y-%m-%d")
    state = _read_json("last_run.json", {})
    if state.get("date") == today:
        return
    hist = _read_json("history.json", [])
    if any(h.get("date") == today for h in hist if isinstance(h, dict)):
        logger.info("今天的日报已发布过，跳过")
        return
    _write_json("last_run.json", {"date": today})
    from junjun_skills.plugins.daily_report import graph as dr_graph
    logger.info(f"热点日报开跑: dr-{today}")
    await dr_graph.run(f"dr-{today}", today)


@register_command("daily_report", aliases=["日报"], plugin="daily_report",
                  description="立即生成今天的热点日报（走审批）")
async def daily_report_cmd(ctx) -> str:
    """手动触发一单（调试用，同样走人审）。"""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    hist = _read_json("history.json", [])
    if any(h.get("date") == today for h in hist if isinstance(h, dict)):
        return "今天的日报已经发过了，明天再来。"
    from junjun_skills.plugins.daily_report import graph as dr_graph
    _write_json("last_run.json", {"date": today})
    state = await dr_graph.run(f"dr-{today}", today)
    if state.get("tid"):
        return "日报已发到空间。"
    if state.get("skip_reason"):
        return f"今天不出日报：{state['skip_reason']}"
    if state.get("error"):
        return f"日报失败了：{state['error']}"
    return "日报写好啦，已发给管理员审批，放行就发空间。"


TOOLS = []

scheduler.add(ScheduledTask("daily_report", daily_report_tick, interval=60,
                            plugin="daily_report"))
