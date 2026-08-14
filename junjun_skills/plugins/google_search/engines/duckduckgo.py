from typing import List, Dict, Any, Optional
import asyncio
import logging

try:
    # 导入新库
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, TimeoutException
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    DDGSException = Exception
    TimeoutException = Exception

from .base import BaseSearchEngine, SearchResult

logger = logging.getLogger(__name__)

def sync_ddgs_search(query: str, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """在一个同步函数中执行 DDGS 文本搜索，以便在线程池中运行

    Args:
        query: 搜索查询
        search_params: 搜索参数字典

    Returns:
        搜索结果字典列表
    """
    timeout = search_params.pop('timeout', 10)
    # proxy 必须传进 DDGS 客户端——2026-08-14 实锤：config 里的 proxy 被
    # 构造器静默吞掉，ddgs 裸连，google/brave/wikipedia 全被墙到超时
    proxy = search_params.pop('proxy', None)
    with DDGS(proxy=proxy, timeout=timeout) as ddgs:
        return ddgs.text(query, **search_params)

def sync_ddgs_images_search(query: str, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """在一个同步函数中执行 DDGS 图片搜索，以便在线程池中运行

    Args:
        query: 搜索查询
        search_params: 搜索参数字典

    Returns:
        图片结果字典列表
    """
    timeout = search_params.pop('timeout', 10)
    proxy = search_params.pop('proxy', None)
    with DDGS(proxy=proxy, timeout=timeout) as ddgs:
        return ddgs.images(query, **search_params)

def _is_no_results(e: Exception) -> bool:
    """「没搜到」是正常空结果，不是故障——不触发直连降级重试。"""
    return isinstance(e, DDGSException) and "No results found" in str(e)

class DuckDuckGoEngine(BaseSearchEngine):
    """使用新版 ddgs 库的搜索引擎实现

    这个库现在是一个元搜索引擎，可以调用多个后端。
    """

    region: str
    backend: str
    safesearch: str
    timelimit: Optional[str]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        if not HAS_DDGS:
            raise ImportError("没有 ddgs 库。请确保它已在插件依赖中声明。")

        # 优化默认配置以提高搜索成功率
        self.region = self.config.get("region", "wt-wt")  # 全球搜索
        self.backend = self.config.get("backend", "auto")  # 自动选择最佳后端
        self.safesearch = self.config.get("safesearch", "moderate")  # 适中的安全搜索
        self.timelimit = self.config.get("timelimit")  # 时间限制，默认为 None
        # 2026-08-14 修复：上层 _build_engine 注入的 proxy 此前被静默吞掉，
        # ddgs 裸连导致墙外引擎全超时（wikipedia/brave/google 连环 Timeout）
        self.proxy = self.config.get("proxy") or None
        # 代理挂掉时的直连降级引擎组：只留国内实测可达的（2026-08-14 生产
        # 日志：yandex 200、duckduckgo 部分可用、bing 国内直连；auto 会让
        # google/brave/wikipedia 等墙外引擎逐个烧超时）
        self.direct_backend = self.config.get("direct_backend", "duckduckgo,bing,yandex")
        self.direct_backend_images = self.config.get(
            "direct_backend_images", "duckduckgo,bing")

        logger.info(f"DuckDuckGo 引擎初始化完成 - region: {self.region}, backend: {self.backend}, safesearch: {self.safesearch}")

    async def _run_text(self, query: str, num_results: int,
                        *, use_proxy: bool) -> List[SearchResult]:
        """跑一轮文本搜索；异常原样抛出，由 search() 决定降级/记录。"""
        loop = asyncio.get_event_loop()
        search_params = {
            'max_results': num_results,
            'region': self.region,
            'backend': self.backend if use_proxy else self.direct_backend,
            'safesearch': self.safesearch,
            'timeout': self.config.get('timeout', 10)
        }
        if self.timelimit:
            search_params['timelimit'] = self.timelimit
        if use_proxy and self.proxy:
            search_params['proxy'] = self.proxy

        search_results = await loop.run_in_executor(
            None, sync_ddgs_search, query, search_params)
        return [SearchResult(
            title=r.get('title', ''), url=r.get('href', ''),
            snippet=r.get('body', ''), abstract=r.get('body', ''), rank=i)
            for i, r in enumerate(search_results)]

    async def _run_images(self, query: str, num_results: int,
                          *, use_proxy: bool) -> List[Dict[str, str]]:
        """跑一轮图片搜索；异常原样抛出，由 search_images() 决定降级/记录。"""
        loop = asyncio.get_event_loop()
        search_params = {
            'max_results': num_results,
            'region': self.region,
            'safesearch': self.safesearch,
            'timeout': self.config.get('timeout', 10)
        }
        if not use_proxy:
            search_params['backend'] = self.direct_backend_images
        if self.timelimit:
            search_params['timelimit'] = self.timelimit
        if use_proxy and self.proxy:
            search_params['proxy'] = self.proxy

        return await loop.run_in_executor(
            None, sync_ddgs_images_search, query, search_params)

    def _log_failure(self, kind: str, query: str, e: Exception) -> None:
        if _is_no_results(e):
            logger.info(f"ddgs {kind}搜索未找到结果: {query}")
        elif isinstance(e, TimeoutException):
            logger.warning(f"ddgs {kind}搜索超时: {query} - {e}")
        elif isinstance(e, DDGSException):
            logger.error(f"ddgs {kind}搜索错误: {e}")
        else:
            logger.error(f"ddgs {kind}搜索意外错误: {query} - {e}", exc_info=True)

    async def search(self, query: str, num_results: int) -> List[SearchResult]:
        """通过在线程池中运行同步的 ddgs.text 方法来进行搜索

        Args:
            query: 搜索查询
            num_results: 期望的结果数量

        Returns:
            搜索结果列表
        """
        try:
            return await self._run_text(query, num_results, use_proxy=True)
        except Exception as e:
            # 代理路径失败（代理间歇性挂是常态，2026-08-14 用户实锤 7890 拒连）
            # 且不是「正常空结果」时，降级直连重试一轮——有代理但代理死了
            # 等于全引擎硬失败，比裸连还差
            if self.proxy and not _is_no_results(e):
                logger.warning(
                    f"ddgs 走代理失败（{type(e).__name__}: {e}），降级直连重试: {query}")
                try:
                    return await self._run_text(query, num_results, use_proxy=False)
                except Exception as e2:
                    self._log_failure("文本", query, e2)
                    return []
            self._log_failure("文本", query, e)
            return []

    async def search_images(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """通过在线程池中运行同步的 ddgs.images 方法来进行图片搜索

        Args:
            query: 搜索查询
            num_results: 期望的结果数量

        Returns:
            图片信息字典列表
        """
        try:
            return await self._run_images(query, num_results, use_proxy=True)
        except Exception as e:
            if self.proxy and not _is_no_results(e):
                logger.warning(
                    f"ddgs 走代理失败（{type(e).__name__}: {e}），降级直连重试(图片): {query}")
                try:
                    return await self._run_images(query, num_results, use_proxy=False)
                except Exception as e2:
                    self._log_failure("图片", query, e2)
                    return []
            self._log_failure("图片", query, e)
            return []
