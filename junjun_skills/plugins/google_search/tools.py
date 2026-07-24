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
        except Exception as exc:
            logger.warning(f"Engine '{engine_name}' failed for query '{query}': {exc}")
            continue

    logger.error(f"All search engines failed for query: {query}")
    return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
async def web_search(query: str, num_results: int = 10) -> str:
    """上网搜索实时信息。当用户问「帮我搜一下」「查一下」「XXX 是什么」「XXX 什么时候」、
    需要最新新闻/资料/时间敏感信息时使用（区别于 search_knowledge 查本地知识库）。

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
