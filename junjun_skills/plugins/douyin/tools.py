"""douyin 插件：抖音分享链接自动/手动解析（迁移自 douyin_video_plugin，新架构重写）。

拦截器：消息含 v.douyin.com 短链或 douyin.com/video|note 链接时自动解析（群聊不限制 @）
命令：/douyin <链接>、/抖音解析 <链接>
API：星知阁 https://api.xingzhige.com/API/douyin/（GET/POST，参数 url=分享链接；
     可用 DOUYIN_API_BASE 覆盖；短链先跟随重定向展开）
响应：兼容 {data:{item,stat}} / 顶层 item/stat / jx[0] 多层结构
发送：摘要文本（标题/作者/点赞/评论/收藏/分享）+ 图集 image 段（上限 9 张）；
     视频默认只发直链文本（QQ 对第三方视频 URL 直发成功率低，与旧插件默认一致）
限流：每会话 30 秒最小间隔（内存 dict）
"""

import os
import re
import time

import httpx

from junjun_agent.commands import register_command
from junjun_agent.interceptors import register_interceptor
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.douyin")

# 常见抖音分享/详情链接（短链 path 常含 - _，如 v.douyin.com/UW8-u_REUP8/）
DOUYIN_URL_RE = re.compile(
    r"https?://(?:"
    r"v\.douyin\.com/[A-Za-z0-9._~-]+/?|"
    r"(?:www\.)?douyin\.com/video/\d+[^ \t\n\r\f\v​]*|"
    r"(?:www\.)?douyin\.com/note/[A-Za-z0-9._~-]+[^ \t\n\r\f\v​]*"
    r")",
    re.IGNORECASE,
)

_API_BASE = os.environ.get("DOUYIN_API_BASE", "https://api.xingzhige.com/API/douyin/")
_TIMEOUT = float(os.environ.get("DOUYIN_API_TIMEOUT", "60"))
_RETRIES = 2                      # 失败重试次数（不含首次），最后一次尝试 POST
_MAX_GALLERY = int(os.environ.get("DOUYIN_MAX_GALLERY", "9"))
_MIN_INTERVAL = float(os.environ.get("DOUYIN_MIN_INTERVAL", "30"))  # 每会话最小间隔（秒）

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 JunJun-Douyin/1.0",
    "Accept": "application/json,text/plain,*/*",
}

# 每会话上次解析时间戳（chat_id -> ts）
_last_use: dict = {}


def _first_douyin_url(text: str) -> str | None:
    """从消息文本提取第一条抖音链接，去掉末尾误匹配的中英文标点。"""
    if not text:
        return None
    m = DOUYIN_URL_RE.search(text)
    return m.group(0).rstrip(".,;!?，。；！？）】>") if m else None


async def _expand_short_url(url: str) -> str:
    """展开 v.douyin.com 短链为落地页 URL（部分解析站对短链报「资源id获取失败」）。

    失败时返回原链接，绝不抛异常。
    """
    if "v.douyin.com/" not in url.lower():
        return url
    try:
        async with httpx.AsyncClient(
            headers=_HTTP_HEADERS, timeout=_TIMEOUT, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            final = str(resp.url)
            if "douyin.com" in final and "v.douyin.com" not in final:
                # 去掉查询串里易过期的 token，保留主路径
                return final.split("?", 1)[0].split("#", 1)[0]
    except Exception as e:
        logger.debug(f"抖音短链展开失败，仍用原链请求解析接口: {type(e).__name__}: {e}")
    return url


async def _fetch_parse(share_url: str) -> dict | None:
    """请求星知阁解析接口，返回原始 JSON；任何失败返回 None（独立 helper 便于测试 mock）。

    流程：短链先展开 -> GET ?url=... 重试 _RETRIES 次 -> 最后一次改用 POST 表单。
    """
    url = await _expand_short_url(share_url)
    api = _API_BASE.rstrip("/") + "/"
    attempts = _RETRIES + 1
    try:
        async with httpx.AsyncClient(headers=_HTTP_HEADERS, timeout=_TIMEOUT) as client:
            for attempt in range(attempts):
                use_post = attempt == attempts - 1 and attempts > 1
                try:
                    if use_post:
                        resp = await client.post(api, data={"url": url})
                    else:
                        resp = await client.get(api, params={"url": url})
                    return resp.json()
                except Exception as e:
                    logger.warning(
                        f"抖音解析接口请求失败 ({attempt + 1}/{attempts}): "
                        f"{type(e).__name__}: {e}"
                    )
    except Exception as e:
        logger.warning(f"抖音解析接口异常: {type(e).__name__}: {e}")
    return None


def _unwrap_payload(root: dict) -> dict:
    """兼容 {data:{item,stat}} 与顶层即 item/stat，最多下探两层 data。"""
    if not isinstance(root, dict):
        return {}
    if "item" in root or "stat" in root or "jx" in root:
        return root
    inner = root.get("data")
    if isinstance(inner, dict):
        if "item" in inner or "stat" in inner or "jx" in inner:
            return inner
        deeper = inner.get("data")
        if isinstance(deeper, dict):
            return deeper
    return root


def _image_urls_from_item(item: dict) -> list:
    """从 item.images 提取图集 URL（兼容 str 或 {url/src/image} 元素）。"""
    raw = item.get("images")
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw:
        if isinstance(x, str) and x.startswith("http"):
            out.append(x)
        elif isinstance(x, dict):
            u = x.get("url") or x.get("src") or x.get("image")
            if isinstance(u, str) and u.startswith("http"):
                out.append(u)
    return out


def _resolve_item_stat(inner: dict) -> tuple:
    """从根对象或 jx[0] 等位置取出 (stat, item)。"""
    stat = inner.get("stat") if isinstance(inner.get("stat"), dict) else {}
    item = inner.get("item") if isinstance(inner.get("item"), dict) else {}
    if item:
        return stat, item
    jx = inner.get("jx")
    if isinstance(jx, list):
        for entry in jx:
            if not isinstance(entry, dict):
                continue
            sub_item = entry.get("item")
            if isinstance(sub_item, dict):
                sub_stat = entry.get("stat") if isinstance(entry.get("stat"), dict) else stat
                return sub_stat, sub_item
            if entry.get("url") or _image_urls_from_item(entry):
                sub_stat = entry.get("stat") if isinstance(entry.get("stat"), dict) else stat
                return sub_stat, entry
    return stat, item


def _api_success(root: dict) -> bool:
    """判断解析是否成功；部分镜像站 code 与 msg 不一致，以「解析成功」文案为准。"""
    if not isinstance(root, dict):
        return False
    msg = str(root.get("msg") or root.get("message") or "")
    if msg:
        if "解析失败" in msg or "未能解析" in msg or "无法解析" in msg:
            return False
        if "解析成功" in msg:
            return True
    if root.get("success") is True:
        return True
    if "code" not in root:
        return True
    try:
        return int(root.get("code")) == 200
    except (TypeError, ValueError):
        return str(root.get("code")).strip().lower() in ("200", "ok", "success")


def _has_sendable_media(item: dict) -> bool:
    if not item:
        return False
    u = item.get("url")
    if isinstance(u, str) and u.startswith("http"):
        return True
    return bool(_image_urls_from_item(item))


def _build_summary(stat: dict, item: dict) -> str:
    """摘要文本：标题/作者 + 点赞/评论/收藏/分享（字段对齐旧插件）。"""
    title = (item.get("title") or item.get("desc") or "抖音").strip()
    author = (item.get("author") or item.get("nickname") or "").strip()
    like = stat.get("like", "—")
    comment = stat.get("comment", "—")
    collect = stat.get("collect", "—")
    share = stat.get("share", "—")
    lines = [title]
    if author:
        lines.append(f"作者：{author}")
    lines.append(f"❤️{like}  💬{comment}  ⭐{collect}  ↗️{share}")
    return "\n".join(lines)


def _check_rate_limit(chat_id: str) -> int:
    """返回剩余冷却秒数；0 表示可解析并记录本次时间。"""
    now = time.time()
    last = _last_use.get(chat_id, 0.0)
    remain = _MIN_INTERVAL - (now - last)
    if remain > 0:
        return int(remain) + 1
    _last_use[chat_id] = now
    return 0


async def _parse_and_reply(ctx, share_url: str) -> None:
    """解析并发送结果；所有失败路径降级为友好中文文本，绝不抛异常。"""
    raw = await _fetch_parse(share_url)
    if raw is None:
        await ctx.reply("抖音解析接口暂时没响应，稍后再试试吧。")
        return

    inner = _unwrap_payload(raw)
    stat, item = _resolve_item_stat(inner)
    if not _api_success(raw) and not _has_sendable_media(item):
        msg = raw.get("msg") or raw.get("message") or "未知原因"
        await ctx.reply(f"抖音解析失败了：{msg}")
        return
    if not _has_sendable_media(item):
        await ctx.reply("解析成功了，但没有拿到可发送的视频或图集。")
        return

    summary = _build_summary(stat, item)
    video_url = item.get("url")
    if isinstance(video_url, str) and video_url.startswith("http"):
        # QQ/NapCat 的 video 段通常不接受抖音 CDN 直链，默认改为正文发直链
        await ctx.reply(f"{summary}\n📎 视频：{video_url}")
        return

    images = _image_urls_from_item(item)[:_MAX_GALLERY]
    if images:
        segs = [ReplySegment(type="text", data=summary)]
        segs += [ReplySegment(type="image", data=img) for img in images]
        await ctx.send(segs)
    else:
        await ctx.reply(summary)


@register_command("douyin", aliases=["抖音解析"], plugin="douyin",
                  description="解析抖音分享链接（视频直链/图集）")
async def douyin_cmd(ctx):
    """手动解析：/douyin <链接> 或 /抖音解析 <链接>。"""
    url = (ctx.args or "").strip()
    if not url:
        return "用法：/douyin <抖音链接>  或  /抖音解析 <链接>"
    if not DOUYIN_URL_RE.search(url):
        return "请提供有效的抖音分享链接（如 v.douyin.com 短链或 douyin.com/video 链接）。"

    remain = _check_rate_limit(ctx.session.chat_id)
    if remain > 0:
        return f"解析太频繁啦，{remain} 秒后再试。"

    logger.info(f"抖音解析(命令): url={url[:80]}")
    await _parse_and_reply(ctx, url)
    return None


@register_interceptor(DOUYIN_URL_RE.pattern, name="douyin_link", plugin="douyin")
async def douyin_hit(ctx) -> bool:
    """自动识别消息中的抖音链接并解析（群聊不限制 @）；消费消息不再进 LLM 决策。"""
    url = (ctx.args or "").rstrip(".,;!?，。；！？）】>")
    if not url:
        return False

    remain = _check_rate_limit(ctx.session.chat_id)
    if remain > 0:
        await ctx.reply(f"抖音解析太频繁啦，{remain} 秒后再试。")
        return True

    logger.info(f"抖音解析(自动): url={url[:80]}")
    await _parse_and_reply(ctx, url)
    return True


TOOLS = []
