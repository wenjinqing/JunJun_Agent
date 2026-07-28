"""链接内容感知：消息里的网页链接 -> 正文摘要 -> 注入上下文。

群友发文章/新闻链接时，君君只能看到 URL 字符串，接不上话。
本模块在进 L3 前快速抓取首个链接的标题+正文摘要（≤300 字）注入 memory_block。

- 只抓首个非媒体链接（图片/视频直链、B站/抖音走各自拦截器，跳过）
- 4s 硬超时，失败/超时静默（不阻塞回复）
- 配置 [link_preview].enable 可关
"""

import re
from typing import Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("memory.link_preview")

_URL_RE = re.compile(r"https?://[^\s<>\"'）】]+")
_TIMEOUT = 4.0
_MAX_CHARS = 300

# 已有专属拦截器/无正文价值的域，跳过
_SKIP_DOMAINS = (
    "bilibili.com", "b23.tv", "douyin.com", "ixigua.com",
    "qlogo.cn", "gtimg.cn", "qq.com/emoji",
)
# 媒体文件直链，跳过
_MEDIA_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".m4s", ".mov", ".pdf")

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_WS_RE = re.compile(r"\s+")


def _first_fetchable_url(text: str) -> Optional[str]:
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;!?，。；！？")
        low = url.lower()
        if any(d in low for d in _SKIP_DOMAINS):
            continue
        if any(low.split("?")[0].endswith(ext) for ext in _MEDIA_EXT):
            continue
        return url
    return None


def _extract_summary(html: str, max_chars: int) -> str:
    """标题 + 正文纯文本摘要（粗糙但够用：去脚本/样式/标签/压缩空白）。"""
    title = ""
    m = _TITLE_RE.search(html)
    if m:
        title = _WS_RE.sub(" ", m.group(1)).strip()
    body = _SCRIPT_RE.sub(" ", html)
    body = _TAG_RE.sub(" ", body)
    body = _WS_RE.sub(" ", body).strip()
    text = f"{title}。{body}" if title else body
    return text[:max_chars].strip("。 ") + ("…" if len(text) > max_chars else "")


async def fetch_link_preview(text: str, *, timeout: float = _TIMEOUT,
                             max_chars: int = _MAX_CHARS) -> str:
    """从消息文本找首个可抓链接，返回正文摘要；无可抓/失败返回空串。"""
    cfg = get_global_config().raw.get("link_preview", {})
    if not cfg.get("enable", True):
        return ""
    url = _first_fetchable_url(text)
    if not url:
        return ""
    try:
        import asyncio

        import httpx
        async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                trust_env=False) as client:
            resp = await asyncio.wait_for(client.get(url), timeout=timeout)
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and "text/plain" not in ctype:
            return ""
        summary = _extract_summary(resp.text, max_chars)
        if summary:
            logger.info(f"链接预览: {url[:60]} -> {summary[:40]}")
        return summary
    except Exception as e:
        logger.debug(f"链接预览失败（忽略）: {url[:60]} {type(e).__name__}")
        return ""
