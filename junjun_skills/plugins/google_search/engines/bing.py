import json
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode, urlparse
from bs4 import BeautifulSoup

from .base import BaseSearchEngine, SearchResult

logger = logging.getLogger(__name__)


class BingEngine(BaseSearchEngine):
    """Bing 搜索引擎实现"""

    base_urls: List[str]
    region: str
    setlang: str
    count: int

    SELECTOR_CONFIG: Dict[str, Dict[str, Any]] = {
        "url": {
            "primary": "h2 > a",
            "fallback": [
                "h2 a",
                "h3 > a",
                ".b_algo h2 a",
                ".b_algo a[href]",
            ],
        },
        "title": {
            "primary": "h2 > a",
            "fallback": [
                "h2 a",
                "h3 > a",
                ".b_algo h2 a",
                ".b_algo a[href]",
            ],
        },
        "text": {
            "primary": ".b_caption > p",
            "fallback": [
                ".b_caption",
                ".b_descript",
                ".b_snippet",
                ".b_algo .b_caption",
            ],
        },
        "links": {
            "primary": "ol#b_results > li.b_algo",
            "fallback": [
                "#b_results > li.b_algo",
                "#b_results li.b_algo",
                ".b_algo",
                "li.b_algo",
            ],
        },
        "next": {
            "primary": 'div#b_content nav[role="navigation"] a.sb_pagN',
            "fallback": [
                'nav[role="navigation"] a.sb_pagN',
                'a.sb_pagN',
                '.sb_pagN',
            ],
        },
    }
    # 黑名单留空，按需添加
    BLOCKED_DOMAINS: List[str] = []

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.base_urls = ["https://cn.bing.com", "https://www.bing.com"]
        self.region = self.config.get("region", "zh-CN")
        self.setlang = self.config.get("setlang", "zh")
        self.count = self.config.get("count", 10)

    def _build_keywords(self, query: str) -> List[str]:
        """构建用于相关性过滤的关键词列表，兼容中英文。"""
        if not query:
            return []
        pieces: List[str] = []
        for token in re.split(r"\s+", query.lower().strip()):
            if not token:
                continue
            words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", token)
            if words:
                pieces.extend(words)
            else:
                pieces.append(token)
        # \u957f\u4e2d\u6587\u4e32\uff08\u22655 \u5b57\uff09\u53e0\u52a0 bigram\uff082026-08-09 \u5b9e\u9524\uff1a\u300c\u5b81\u5fb7\u5230\u6df1\u5733\u300d\u6574\u4e32
        # \u5f53\u4e00\u4e2a\u5173\u952e\u8bcd\uff0c\u7ed3\u679c\u5199\u300c\u5b81\u5fb7\u81f3\u6df1\u5733\u300d\u300c\u5b81\u5fb7\u5230\u6df1\u5733\u5317\u300d\u5c31\u96f6\u547d\u4e2d\uff0c5 \u4e2a
        # \u67e5\u8be2 3 \u4e2a\u88ab\u8fc7\u6ee4\u6e05\u96f6\uff09\u2014\u2014bigram \u8ba9\u57ce\u5e02/\u4e8b\u7269\u540d\u72ec\u7acb\u547d\u4e2d\u3002\u4f1a\u4ea7\u751f\u5c11\u91cf
        # \u566a\u97f3\u7247\u6bb5\uff08\u300c\u5fb7\u5230\u300d\uff09\uff0c\u4f46\u8fc7\u6ee4\u8bed\u4e49\u662f\u300c\u81f3\u5c11\u547d\u4e2d\u4e00\u4e2a\u300d\uff0c\u566a\u97f3\u65b9\u5411\u662f
        # \u653e\u8fc7\u4e0d\u662f\u9519\u6740\uff0c\u53ef\u63a5\u53d7\u3002
        out: List[str] = []
        for p in pieces:
            if len(p) < 2:
                continue
            out.append(p)
            if re.fullmatch(r"[\u4e00-\u9fff]{5,}", p):
                out.extend(p[i:i + 2] for i in range(len(p) - 1))
        return out

    def _is_relevant(self, title: str, snippet: str, url: str, keywords: List[str]) -> bool:
        """粗粒度过滤：标题/摘要/URL 命中至少一个关键词即通过。"""
        if not keywords:
            return True
        text = f"{title} {snippet} {url}".lower()
        match_count = sum(1 for kw in keywords if kw in text)
        return match_count >= 1

    def _is_blocked(self, url: str) -> bool:
        """域名黑名单过滤。"""
        if not url:
            return False
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            return False
        return any(netloc.endswith(domain) for domain in self.BLOCKED_DOMAINS)

    def _set_selector(self, selector: str) -> str:
        """获取页面元素选择器。"""
        config = self.SELECTOR_CONFIG.get(selector, {})
        return config.get("primary", "")

    def _get_fallback_selectors(self, selector: str) -> list:
        """获取备用选择器列表。"""
        config = self.SELECTOR_CONFIG.get(selector, {})
        return config.get("fallback", [])

    async def _get_next_page(
        self,
        query: str,
        *,
        base_url: Optional[str] = None,
        region: Optional[str] = None,
        setlang: Optional[str] = None,
        market: Optional[str] = None,
    ) -> str:
        """构建并获取搜索页面的 HTML 内容。

        Args:
            query: 搜索查询词。
            base_url: 基础域名，默认先用 cn，再回退 www。
            region: 区域代码（cc），影响市场。
            setlang: 语言参数。
            market: 市场参数（mkt），可用于强制 en-US 等。

        Returns:
            HTML 文本。
        """
        base_url = base_url or self.base_urls[0]
        region = region or self.region
        setlang = setlang or self.setlang
        params = {
            "q": query,
            "setlang": setlang,
            "count": str(min(self.count, 50)),
            "ensearch": "1",
        }
        if region:
            params["cc"] = region.split("-")[0] if "-" in region else region
        if market:
            params["mkt"] = market

        query_string = urlencode(params)
        search_url = f"{base_url}/search?{query_string}"
        logger.info(f"Requesting Bing search URL: {search_url}")
        return await self._get_html(search_url)

    def _get_link_elements(self, soup: BeautifulSoup) -> List[Any]:
        """获取搜索结果节点，包含主选择器和回退。"""
        links_selector = self._set_selector("links")
        if links_selector:
            links = soup.select(links_selector)
            if links:
                return links
        for fallback_selector in self._get_fallback_selectors("links"):
            links = soup.select(fallback_selector)
            if links:
                logger.info(f"Fallback selector '{fallback_selector}' found {len(links)} results")
                return links
        return []

    def _select_with_fallback(self, element: Any, selector_name: str) -> Optional[Any]:
        """在元素上按主/备用选择器查找单个子元素。"""
        primary = self._set_selector(selector_name)
        if primary:
            found = element.select_one(primary)
            if found:
                return found
        for fallback in self._get_fallback_selectors(selector_name):
            found = element.select_one(fallback)
            if found:
                return found
        return None

    def _parse_page_results(self, soup: BeautifulSoup, keywords: List[str]) -> List[SearchResult]:
        """解析页面并生成过滤后的 SearchResult 列表。"""
        links = self._get_link_elements(soup)
        if not links:
            return []

        results: List[SearchResult] = []
        for idx, link in enumerate(links):
            title_elem = self._select_with_fallback(link, "title")
            url_elem = self._select_with_fallback(link, "url")
            text_elem = self._select_with_fallback(link, "text")

            title = self.tidy_text(title_elem.text) if title_elem else ""
            url_raw = url_elem.get("href") if url_elem else ""
            url = self._normalize_url(url_raw)
            snippet = self.tidy_text(text_elem.text) if text_elem else ""

            if title and url and not self._is_blocked(url) and self._is_relevant(title, snippet, url, keywords):
                results.append(SearchResult(title=title, url=url, snippet=snippet, abstract=snippet, rank=idx))
        return results

    def _parse_page_unfiltered(self, soup: BeautifulSoup) -> List[SearchResult]:
        """不过滤的解析（过滤清零时的保守回退——宁可收噪音，不可错杀）。"""
        links = self._get_link_elements(soup)
        results: List[SearchResult] = []
        for idx, link in enumerate(links):
            title_elem = self._select_with_fallback(link, "title")
            url_elem = self._select_with_fallback(link, "url")
            text_elem = self._select_with_fallback(link, "text")
            title = self.tidy_text(title_elem.text) if title_elem else ""
            url = self._normalize_url(url_elem.get("href") if url_elem else "")
            snippet = self.tidy_text(text_elem.text) if text_elem else ""
            if title and url and not self._is_blocked(url):
                results.append(SearchResult(title=title, url=url, snippet=snippet, abstract=snippet, rank=idx))
        return results

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        """执行搜索，使用多区域尝试与选择器回退。"""
        try:
            keywords = self._build_keywords(query)
            fetch_variants = [
                {
                    "base_url": self.base_urls[0],
                    "region": self.region,
                    "setlang": self.setlang,
                    "market": self.region,
                },
            ]
            # 纯中文查询不试 en-US 变体（中文词在英文市场返回的多是无关结果，
            # 白烧一次请求——2026-08-09 日志实锤三个中文查询空转 en 变体）
            if re.search(r"[a-zA-Z]", query):
                fetch_variants.append({
                    "base_url": self.base_urls[1] if len(self.base_urls) > 1 else self.base_urls[0],
                    "region": "en-US",
                    "setlang": "en",
                    "market": "en-US",
                })

            results: List[SearchResult] = []
            for variant in fetch_variants:
                resp = await self._get_next_page(query, **variant)
                soup = BeautifulSoup(resp, "html.parser")
                page_results = self._parse_page_results(soup, keywords)
                if page_results:
                    results.extend(page_results)
                    break

            if not results:
                logger.warning(f"No relevant results remain after filtering for query '{query}'")

            logger.info(f"Returning {len(results[:num_results])} search results for query '{query}'")
            return results[:num_results]
        except Exception as e:
            logger.error(f"Error in Bing search for query {query}: {e}", exc_info=True)
            return []

    async def search_unfiltered(self, query: str, num_results: int) -> List[SearchResult]:
        """无过滤裸搜索：fallback 链全空后的最后手段（由调用方决定是否使用）——
        相关性过滤是防噪音的不是门禁，但无过滤结果不该在链内截胡后面引擎的
        过滤结果（2026-08-09 实测：裸结果截胡导致 DNS 问答顶掉交通查询）。"""
        try:
            resp = await self._get_next_page(
                query, base_url=self.base_urls[0],
                region=self.region, setlang=self.setlang, market=self.region)
            soup = BeautifulSoup(resp, "html.parser")
            results = self._parse_page_unfiltered(soup)
            logger.info(f"未过滤兜底返回 {len(results[:num_results])} 条: '{query}'")
            return results[:num_results]
        except Exception as e:
            logger.warning(f"未过滤兜底搜索失败 '{query}': {type(e).__name__}: {e}")
            return []

    async def search_images(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """执行Bing图片搜索（国内可直接访问，无需科学上网）

        Args:
            query: 搜索关键词
            num_results: 期望的图片数量

        Returns:
            图片信息字典列表，格式：[{"image": "图片URL", "title": "图片标题", "thumbnail": "缩略图URL"}]
        """
        try:
            params = {
                "q": query,
                "first": 1,
                "count": min(num_results, 150),
                "cw": 1177,
                "ch": 826,
                "FORM": "HDRSC2"
            }

            # 尝试多个Bing图片搜索域名
            html = ""
            successful_base_url = ""
            for base_url in self.base_urls:
                try:
                    search_url = f"{base_url}/images/search?{urlencode(params)}"
                    logger.debug(f"请求Bing图片搜索URL: {search_url}")
                    html = await self._get_html(search_url)
                    if html and ("img_cont" in html or "iusc" in html):
                        successful_base_url = base_url
                        break
                except Exception as e:
                    logger.warning(f"Bing图片搜索域名 {base_url} 失败: {e}")
                    continue

            if not html:
                logger.warning(f"Bing图片搜索未获取到有效HTML: {query}")
                return []

            soup = BeautifulSoup(html, "html.parser")
            results = []

            # 解析图片结果 - Bing图片搜索的HTML结构
            image_elements = soup.select("a.iusc")

            for elem in image_elements[:num_results]:
                try:
                    # 尝试从m属性获取JSON数据
                    m_attr = elem.get("m")
                    if m_attr:
                        try:
                            m_data = json.loads(m_attr)
                            image_url = m_data.get("murl", "")
                            thumbnail_url = m_data.get("turl", "")
                            title = m_data.get("t", "")

                            if image_url and image_url.startswith(("http://", "https://")):
                                results.append({
                                    "image": image_url,
                                    "title": title or query,
                                    "thumbnail": thumbnail_url or image_url
                                })
                                continue
                        except json.JSONDecodeError:
                            # JSON解析失败，尝试备用方法
                            pass

                    # 备用解析：从img标签获取
                    img_elem = elem.find("img")
                    if img_elem:
                        image_url = img_elem.get("src") or img_elem.get("data-src")
                        if image_url:
                            # 处理相对路径
                            if image_url.startswith("//"):
                                image_url = "https:" + image_url
                            elif image_url.startswith("/") and successful_base_url:
                                image_url = f"{successful_base_url}{image_url}"

                            if image_url.startswith(("http://", "https://")):
                                title = img_elem.get("alt") or query
                                results.append({
                                    "image": image_url,
                                    "title": title,
                                    "thumbnail": image_url
                                })
                except Exception as e:
                    logger.debug(f"解析Bing图片元素失败: {e}")
                    continue

            logger.debug(f"Bing图片搜索找到 {len(results)} 张图片: {query}")
            return results[:num_results]

        except Exception as e:
            logger.error(f"Bing图片搜索错误: {query} - {e}", exc_info=True)
            return []
