"""Google/Bing/Sogou/DuckDuckGo/Tavily/You multi-engine search + abbreviation translation tools."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain.tools import tool

from junjun_core.observability import get_logger

from .engines.bing import BingEngine
from .engines.duckduckgo import DuckDuckGoEngine
from .engines.google import GoogleEngine
from .engines.sogou import SogouEngine
from .engines.tavily import TavilyEngine
from .engines.you import YouSearchEngine
from .translators.nbnhhsh import NbnhhshTranslator

logger = get_logger("google_search.tools")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


DEFAULT_ENGINE: str = _env("GOOGLE_SEARCH_DEFAULT_ENGINE", "google")
DEFAULT_NUM_RESULTS: int = int(_env("GOOGLE_SEARCH_DEFAULT_NUM_RESULTS", "10"))

TAVILY_API_KEY: str = _env("TAVILY_API_KEY", "")
YOU_API_KEY: str = _env("YOU_API_KEY", "")

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------
ENGINE_MAP: Dict[str, Any] = {
    "google": GoogleEngine,
    "bing": BingEngine,
    "sogou": SogouEngine,
    "duckduckgo": DuckDuckGoEngine,
    "tavily": TavilyEngine,
    "you": YouSearchEngine,
}

ENGINE_PRIORITY: List[str] = ["google", "bing", "sogou", "duckduckgo", "tavily", "you"]

# 缺可选依赖的引擎（ImportError）：本轮会话永久跳过，不每查询空转刷告警
_PERMANENTLY_BROKEN: set = set()


def _build_engine(name: str) -> Any:
    """Instantiate a search engine by name with optional API keys."""
    config: Dict[str, Any] = {}
    if name == "tavily" and TAVILY_API_KEY:
        config["api_keys"] = [TAVILY_API_KEY]
    if name == "you" and YOU_API_KEY:
        config["api_keys"] = [YOU_API_KEY]
    return ENGINE_MAP[name](config=config)


async def _search_with_fallback(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """Try engines in priority order until one returns results."""
    # Preferred engine first
    preferred = DEFAULT_ENGINE.lower()
    order = [preferred] + [e for e in ENGINE_PRIORITY if e != preferred]

    for engine_name in order:
        if engine_name in _PERMANENTLY_BROKEN:
            continue  # 缺依赖的引擎不再每次查询空转一轮（2026-08-09：google 缺
            # googlesearch-python，5 个查询刷 5 条 warning 噪音）
        engine_cls = ENGINE_MAP.get(engine_name)
        if engine_cls is None:
            continue
        try:
            engine = _build_engine(engine_name)
            results = await engine.search(query, num_results)
            if results:
                logger.info(
                    f"Search succeeded using engine='{engine_name}' query='{query}' results={len(results)}"
                )
                return [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "abstract": r.abstract,
                        "rank": r.rank,
                        "content": r.content,
                    }
                    for r in results
                ]
        except ImportError as exc:
            _PERMANENTLY_BROKEN.add(engine_name)
            logger.warning(f"Engine '{engine_name}' 依赖缺失，本轮会话永久跳过: {exc}")
            continue
        except Exception as exc:
            logger.warning(f"Engine '{engine_name}' failed for query '{query}': {exc}")
            continue

    # 所有引擎的过滤结果都空 -> 最后手段：bing 无过滤裸结果
    # （相关性过滤是防噪音的不是门禁；宁收噪音不交白卷——但只在链尾，
    # 不让裸结果截胡后面引擎的过滤结果）
    try:
        engine = _build_engine("bing")
        results = await engine.search_unfiltered(query, num_results)
        if results:
            logger.info(f"全链过滤为空，使用 bing 未过滤兜底（{len(results)} 条）: '{query}'")
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "abstract": r.abstract,
                    "rank": r.rank,
                    "content": r.content,
                }
                for r in results
            ]
    except Exception as exc:
        logger.warning(f"未过滤兜底失败 '{query}': {exc}")

    logger.error(f"All search engines failed for query: {query}")
    return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
async def web_search(query: str, num_results: int = 10) -> str:
    """上网搜索实时信息。当用户问「帮我搜一下」「查一下」「XXX 是什么」「XXX 什么时候」、
    需要最新新闻/资料/时间敏感信息时使用（区别于 search_knowledge 查本地知识库）。

    【query 怎么写】用搜索关键词风格，不要把用户的原话拆块填进去：
    - 短、具体、名词/专名为主（3-6 个词），去掉「谁/怎么/为什么/告诉我」等口语壳
    - 坏例：「斐波那契数列通项公式 推导历史 谁推导出来的」（原话拆块，引擎匹配差）
    - 好例：「斐波那契 通项公式 Binet 推导」
    - 一次没搜到就换同义词/换角度再搜，不要原样重试；最多换 2 次角度，
      仍不理想就用已有结果如实回答（说明信息有限），不要无限重搜

    Args:
        query: 搜索关键词
        num_results: 返回结果数上限（默认 10）

    Returns:
        JSON 格式的搜索结果（标题/链接/摘要）。
    """
    import json as _json

    _num = min(num_results, DEFAULT_NUM_RESULTS) if DEFAULT_NUM_RESULTS else num_results
    results = await _search_with_fallback(query, _num)
    return _json.dumps(results, ensure_ascii=False, indent=2)


@tool
async def abbreviation_translate(term: str) -> str:
    """Translate an internet abbreviation/ slang (e.g., 'yyds', 'xswl') using Nbnhhsh.

    Args:
        term: The abbreviation or slang to translate.

    Returns:
        A JSON-formatted string with translations or an empty list.
    """
    import json as _json

    translator = NbnhhshTranslator()
    result = await translator.translate(term)
    payload = {
        "query": result.query,
        "translations": result.translations,
        "source": result.source,
    }
    return _json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Export for plugin_loader
# ---------------------------------------------------------------------------
TOOLS = [web_search, abbreviation_translate]
