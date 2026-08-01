"""深度研究 job：规划 -> 多路搜索 -> 读全文 -> 综述（结构化流水线，非 agent 循环）。

与 agent_task 的分工：
- agent_task：通用隔离子 agent，模型自己打转，结果不可控
- deep_research：代码控制流水线的「深度」版——拆查询、多源检索、读原文、
  交叉验证、写带来源的报告。每一步可观测可测试，不受弱模型打转影响

搜索复用 google_search 插件的多引擎 fallback（google/bing/sogou/duckduckgo/
tavily 轮着试）；读全文用 MCP fetch 工具（没有则降级为摘要级研究，报告
依然可写，只是浅）。全链路任何单步失败都降级，不把整个 job 炸掉。
"""

import asyncio
import json
import re
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.deep_research")

_PLAN_PROMPT = """你是研究规划师。把研究主题拆成 {n} 个互补的搜索查询。
覆盖角度：基础概念/现状数据/对比或争议/最新进展（按主题性质取舍）。
主题：{topic}
只输出 JSON 数组：["查询1", "查询2", ...]，不要输出别的。"""

_SYNTH_PROMPT = """你是研究报告撰写者。基于检索材料写一份中文研究报告。
主题：{topic}

检索材料：
{materials}

要求：
- 正文不超过 {max_chars} 字，分 3-5 个小节（每节一句话小标题）
- 只写有材料支撑的内容；材料不足的角度直接承认「没查到」，不许编
- 结尾附「来源」列表（最多 6 条，格式：标题 - URL）
- 直接输出报告全文，不要解释你的过程"""


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("deep_research", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- 检索后端

async def _default_search(query: str, num: int) -> list:
    """多引擎 fallback 搜索 -> [{title,url,snippet}]（失败返回 []）。"""
    from junjun_skills.plugins.google_search.tools import _search_with_fallback
    return await _search_with_fallback(query, num)


def _mcp_fetch_tool():
    """MCP 抓取工具（名字含 fetch），没有则 None。"""
    from junjun_skills.registry import get_tools
    for t in get_tools():
        if t.name.startswith("mcp_") and "fetch" in t.name:
            return t
    return None


async def _default_fetch(url: str, max_chars: int) -> str:
    """MCP 读全文；不可用/失败返回 ""（调用方降级用摘要）。"""
    tool = _mcp_fetch_tool()
    if tool is None:
        return ""
    try:
        str_args = [k for k, v in (tool.args or {}).items()
                    if isinstance(v, dict) and v.get("type") == "string"]
        if not str_args:
            return ""
        out = await tool.ainvoke({str_args[0]: url})
        text = out if isinstance(out, str) else str(out)
        return text[:max_chars]
    except Exception as e:
        logger.debug(f"读全文失败（降级摘要） {url[:60]}: {type(e).__name__}")
        return ""


# ---------------------------------------------------------------- 流水线

def _parse_queries(raw: str, topic: str, n: int) -> list:
    """解析规划输出；任何异常降级为 [主题本身]（流水线不炸）。"""
    m = re.search(r"\[.*\]", raw or "", flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            queries = [str(q).strip()[:80] for q in arr
                       if q is not None and str(q).strip()]
            if queries:
                return queries[:n]
        except (json.JSONDecodeError, ValueError):
            pass
    return [topic]


async def _plan(topic: str, model=None) -> list:
    n = int(_cfg().get("queries", 5))
    try:
        if model is None:
            from junjun_llm import get_chat_model
            model = get_chat_model("utils")
        from langchain_core.messages import HumanMessage
        resp = await model.ainvoke([HumanMessage(content=_PLAN_PROMPT.format(n=n, topic=topic))])
        return _parse_queries(str(resp.content), topic, n)
    except Exception as e:
        logger.warning(f"研究规划失败（降级单查询）: {type(e).__name__}: {e}")
        return [topic]


async def _collect(queries: list, *, search=None, fetch=None) -> list:
    """检索 + 读全文。返回 [{title,url,snippet,content}]，URL 全局去重。"""
    search = search or _default_search
    fetch = fetch or _default_fetch
    pages_per_query = int(_cfg().get("pages_per_query", 2))
    fetch_max = int(_cfg().get("fetch_max_chars", 3000))

    result_lists = await asyncio.gather(
        *(search(q, pages_per_query + 1) for q in queries), return_exceptions=True)
    seen, items = set(), []
    for results in result_lists:
        if isinstance(results, Exception):
            continue
        for r in (results or [])[:pages_per_query]:
            url = str(r.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            items.append({"title": str(r.get("title") or "")[:80], "url": url,
                          "snippet": str(r.get("snippet") or "")[:300], "content": ""})
    if not items:
        return []

    # 读全文（并发 4，单篇失败降级摘要）
    sem = asyncio.Semaphore(4)

    async def _fill(item):
        async with sem:
            item["content"] = await fetch(item["url"], fetch_max)
    await asyncio.gather(*(_fill(it) for it in items), return_exceptions=True)
    got = sum(1 for it in items if it["content"])
    logger.info(f"深度研究检索: {len(queries)} 查询 -> {len(items)} 篇（{got} 篇读到全文）")
    return items


def _fmt_materials(items: list) -> str:
    parts = []
    for i, it in enumerate(items, 1):
        body = it["content"] or it["snippet"]
        parts.append(f"[{i}] {it['title']}\n{it['url']}\n{body}")
    return "\n\n".join(parts)


async def _synthesize(topic: str, items: list, model=None) -> str:
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("agent")
    from langchain_core.messages import HumanMessage
    resp = await model.ainvoke([HumanMessage(content=_SYNTH_PROMPT.format(
        topic=topic, materials=_fmt_materials(items)[:12000],
        max_chars=int(_cfg().get("report_max_chars", 1000))))])
    text = str(resp.content).strip()
    if not text:
        raise RuntimeError("综述模型返回空")
    return text


async def deep_research_handler(job, payload: dict, *, plan_model=None,
                                synth_model=None, search=None, fetch=None) -> str:
    """深度研究主流程（引擎 handler）。抛异常 = job 失败，由队列引擎兜底。"""
    topic = str(payload.get("topic") or job.title).strip()[:200]
    if not topic:
        raise RuntimeError("研究主题为空")
    queries = await _plan(topic, plan_model)
    logger.info(f"深度研究 [{job.job_id}] 主题「{topic[:30]}」拆出 {len(queries)} 个查询")
    items = await _collect(queries, search=search, fetch=fetch)
    if not items:
        raise RuntimeError("所有搜索引擎都没查到东西，换个说法试试")
    return await _synthesize(topic, items, synth_model)
