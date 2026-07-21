"""news 插件：每天 60 秒读懂世界 + 历史上的今天（迁移自 news_plugin，新架构重写）。

命令：/news /新闻、/history /历史
工具：get_today_in_history（LLM 可用，对齐旧插件开放面）
API：60s.viki.moe/v2/60s、60s.viki.moe/v2/today-in-history（可用 NEWS_API_BASE 覆盖）
"""

import os

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.news")

_API_BASE = os.environ.get("NEWS_API_BASE", "https://60s.viki.moe")
_TIMEOUT = 12.0
_MAX_HISTORY = 10


async def _get(path: str):
    """GET JSON，失败返回 None。"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_API_BASE}{path}")
            data = resp.json()
        return data if data.get("code") == 200 else None
    except Exception as e:
        logger.warning(f"news API 请求失败 {path}: {type(e).__name__}: {e}")
        return None


async def fetch_60s_news() -> dict | None:
    """返回 {'news': [...], 'tip': str, 'image': url} 或 None。"""
    data = await _get("/v2/60s")
    if not data or not data.get("data"):
        return None
    d = data["data"]
    return {"news": d.get("news") or [], "tip": d.get("tip") or "",
            "image": d.get("image") or ""}


async def fetch_today_in_history(limit: int = _MAX_HISTORY) -> list | None:
    """返回 ['事件1', ...] 或 None。"""
    data = await _get("/v2/today-in-history")
    if not data or not data.get("data"):
        return None
    items = (data["data"].get("items") or [])[:limit]
    out = []
    for it in items:
        title = it.get("title") or it.get("event") or ""
        year = it.get("year") or ""
        out.append(f"{year}年 {title}" if year else title)
    return out


@register_command("news", aliases=["新闻"], plugin="news", description="每天60秒读懂世界")
async def news_cmd(ctx):
    result = await fetch_60s_news()
    if not result or not result["news"]:
        return "新闻获取失败了，稍后再试试吧。"
    lines = [f"{i}. {n}" for i, n in enumerate(result["news"], 1)]
    text = "📰 每天 60 秒读懂世界：\n" + "\n".join(lines)
    if result["tip"]:
        text += f"\n\n【微语】{result['tip']}"
    if result["image"]:
        await ctx.send([ReplySegment(type="text", data=text),
                        ReplySegment(type="image", data=result["image"])])
        return None
    return text


@register_command("history", aliases=["历史"], plugin="news", description="历史上的今天")
async def history_cmd(ctx):
    items = await fetch_today_in_history()
    if not items:
        return "历史上的今天获取失败了，稍后再试试吧。"
    return "📜 历史上的今天：\n" + "\n".join(f"- {t}" for t in items)


@tool
async def get_today_in_history() -> str:
    """查询"历史上的今天"发生的大事。被问历史上的今天、想聊历史话题时使用。"""
    items = await fetch_today_in_history(limit=6)
    if not items:
        return "查询失败，稍后再试。"
    return "历史上的今天：\n" + "\n".join(f"- {t}" for t in items)


TOOLS = [get_today_in_history]
