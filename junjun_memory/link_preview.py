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
_MAX_BYTES = 512 * 1024  # 响应体读取上限（防超大页面撑爆内存）


def _is_forbidden_target(url: str) -> bool:
    """SSRF 防护：目标解析到私网/回环/保留地址段则拒绝抓取。

    群友可发任意 URL——不过滤的话 bot 会替他们探测内网服务
    （http://192.168.x / 169.254.169.254 云元数据）并把内容注入 prompt。
    DNS 解析失败同样拒绝。注意：这是 best-effort（DNS rebinding 不在防范围）。
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname
        if not host:
            return True
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False

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
    if _is_forbidden_target(url):
        logger.warning(f"链接预览拒绝（SSRF 防护，目标为内网/保留地址）: {url[:60]}")
        return ""
    try:
        import asyncio

        import httpx
        async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                trust_env=False) as client:
            # 流式读取 + 字节上限：超大页面不整个进内存
            async with client.stream("GET", url) as resp:
                ctype = resp.headers.get("content-type", "")
                if "text/html" not in ctype and "text/plain" not in ctype:
                    return ""
                chunks, total = [], 0
                async def _read():
                    nonlocal total
                    async for chunk in resp.aiter_bytes(65536):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= _MAX_BYTES:
                            break
                await asyncio.wait_for(_read(), timeout=timeout)
        content = b"".join(chunks)
        text_body = content.decode("utf-8", errors="replace")
        summary = _extract_summary(text_body, max_chars)
        if summary:
            logger.info(f"链接预览: {url[:60]} -> {summary[:40]}")
        return summary
    except Exception as e:
        logger.debug(f"链接预览失败（忽略）: {url[:60]} {type(e).__name__}")
        return ""
