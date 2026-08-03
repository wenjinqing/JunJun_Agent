"""抖音内容理解：分享链接 -> SSR 数据 -> 材料 -> 摘要 -> 上下文注入 + 理解工具。

与 bilibili/content.py 同构，但数据源是**第一方 SSR**（2026-08-02 实测）：
- 短链 v.douyin.com 302 -> iesdouyin.com/share/video/<id>/（直播跳 webcast，识别后婉拒）
- 分享页用移动端 UA 抓 window._ROUTER_DATA -> videoInfoRes.item_list[0]：
  文案/作者/BGM/互动数据/时长/play_addr 直链，全程无签名无第三方解析站
- 评论 API 需 X-Bogus 签名（逆向活），不碰——抖音短视频文案+BGM 已够聊

材料按 aweme_id 缓存 30 分钟 + in-flight 共享；摘要进会话「最近抖音」清单，
君君像真人一样「瞄过一眼」群里刚分享的抖音，能自然接话。
"""

import asyncio
import json
import re
import time
from collections import deque
from typing import Dict, Optional, Tuple

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.douyin.content")

_CACHE_TTL = 1800.0
_RECENT_TTL = 1800.0
_RECENT_MAX = 3

_MATERIAL_CACHE: Dict[str, Tuple[float, dict]] = {}  # aweme_id -> (ts, material)
_PENDING: Dict[str, asyncio.Task] = {}
_RECENT: Dict[str, deque] = {}

_UA_MOBILE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
              "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
_TIMEOUT = 15.0
_ROUTER_RE = re.compile(r"window\._ROUTER_DATA\s*=\s*(\{.*?\})\s*</script>", re.S)


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("douyin", {}) or {}
    except Exception:
        return {}


def _understand_enabled() -> bool:
    return bool(_cfg().get("enable_understand", True))


# ---------------------------------------------------------------- 解析（实测链路）

def _parse_router_data(body: str) -> Optional[dict]:
    """分享页 HTML -> videoInfoRes.item_list[0]；结构不符返回 None。"""
    m = _ROUTER_RE.search(body or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        for v in (data.get("loaderData") or {}).values():
            res = (v or {}).get("videoInfoRes")
            if res and res.get("item_list"):
                return res["item_list"][0]
    except Exception:
        pass
    return None


def _item_to_info(aweme_id: str, item: dict) -> dict:
    """item_list[0] -> 统一 info dict。"""
    st = item.get("statistics") or {}
    video = item.get("video") or {}
    play_list = ((video.get("play_addr") or {}).get("url_list")) or []
    play_url = (play_list[0] if play_list else "").replace("http://", "https://")
    return {
        "aweme_id": aweme_id,
        "desc": (item.get("desc") or "").strip(),
        "author": ((item.get("author") or {}).get("nickname") or "").strip(),
        "music": ((item.get("music") or {}).get("title") or "").strip(),
        "digg": st.get("digg_count", 0), "comments": st.get("comment_count", 0),
        "shares": st.get("share_count", 0),
        "duration_s": int((video.get("duration") or 0) / 1000),
        "play_url": play_url,
        "page_url": f"https://www.douyin.com/video/{aweme_id}",
    }


async def _resolve_video(url: str) -> Optional[dict]:
    """分享链接 -> info dict；直播返回 {"type":"live"}，失败 None。绝不抛异常。"""
    import httpx
    try:
        async with httpx.AsyncClient(
                headers={"User-Agent": _UA_MOBILE}, timeout=_TIMEOUT,
                follow_redirects=False) as client:
            real_url = url
            if "v.douyin.com" in url:
                resp = await client.get(url)
                loc = resp.headers.get("location", "")
                if not loc:
                    return None
                real_url = loc
            if "webcast" in real_url:
                return {"type": "live"}  # 直播/直播回放：没有 SSR 视频数据
            m = re.search(r"/(?:video|note)/(\d{10,25})", real_url)
            if not m:
                return None
            aweme_id = m.group(1)
            share = f"https://www.iesdouyin.com/share/video/{aweme_id}/"
            resp = await client.get(share, follow_redirects=True)
            item = _parse_router_data(resp.text or "")
            if not item:
                return None
            return _item_to_info(aweme_id, item)
    except Exception as e:
        logger.debug(f"抖音解析失败: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------- 材料构建（缓存 + 共享）

async def _build_material(url: str) -> Optional[dict]:
    info = await _resolve_video(url)
    if not info or info.get("type") == "live":
        return info  # live 原样返回（工具层给出专门话术）
    parts = [f"文案：{info['desc'] or '（无文案）'}",
             f"作者：{info['author'] or '未知'}"]
    if info["music"]:
        parts.append(f"BGM：{info['music']}")
    parts.append(f"时长：{info['duration_s']}秒")
    parts.append(f"互动：{info['digg']}赞/{info['comments']}评/{info['shares']}转发")
    return {"info": info, "material": "\n".join(parts), "source": "文案数据",
            "page_url": info["page_url"]}


async def get_material(url: str) -> Optional[dict]:
    """提取抖音内容材料（30 分钟缓存 + in-flight 共享）。失败返回 None。"""
    from junjun_skills.plugins.douyin.tools import _first_douyin_url
    link = _first_douyin_url(url) or (url if "douyin.com" in (url or "") else None)
    if not link:
        return None
    key = link
    hit = _MATERIAL_CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    task = _PENDING.get(key)
    if task is None or task.done():
        task = asyncio.create_task(_build_material(link))
        _PENDING[key] = task
        task.add_done_callback(lambda _t, k=key: _PENDING.pop(k, None))
    try:
        result = await task
    except Exception as e:
        logger.debug(f"抖音材料构建失败: {type(e).__name__}: {e}")
        return None
    if result and result.get("material"):
        _MATERIAL_CACHE[key] = (time.time(), result)
        logger.info(f"抖音材料: {result['info']['aweme_id']}（{len(result['material'])} 字）")
    return result


async def get_play_url(url: str) -> str:
    """深看用：取视频直链（复用材料缓存）。"""
    m = await get_material(url)
    if m and m.get("info"):
        return m["info"].get("play_url") or ""
    return ""


# ---------------------------------------------------------------- 摘要

_SUM_PROMPT = """基于以下抖音视频的资料，用中文口语一两句话讲清这个视频大概是什么内容/看点
（{max_chars} 字以内；资料只有文案和互动数据，判断不了细节就只说主题，不许编）。
资料：
{material}"""


async def summarize_material(material: str, *, model=None, max_chars: int = 80) -> str:
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
    from langchain_core.messages import HumanMessage
    resp = await model.ainvoke([HumanMessage(content=_SUM_PROMPT.format(
        max_chars=max_chars, material=material[:3000]))])
    return str(resp.content).strip()


# ---------------------------------------------------------------- 预热 + 会话最近抖音

def prewarm_video(chat_id: str, url: str) -> None:
    """后台「看懂」抖音：材料入缓存 + 摘要进会话最近清单。fire-and-forget。"""
    if not _understand_enabled():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_watch(chat_id, url))


async def _watch(chat_id: str, url: str) -> None:
    try:
        m = await get_material(url)
        if not m or not m.get("info"):
            return
        summary = ""
        try:
            summary = await summarize_material(m["material"])
        except Exception:
            pass
        info = m["info"]
        dq = _RECENT.setdefault(chat_id, deque(maxlen=_RECENT_MAX * 2))
        dq.append((time.time(), {
            "title": info["desc"][:40] or "抖音视频",
            "owner": info["author"],
            "summary": summary or info["desc"][:60],
            "page_url": m["page_url"],
        }))
    except Exception as e:
        logger.debug(f"抖音后台理解失败（忽略）: {type(e).__name__}: {e}")


def render_recent_block(chat_id: str) -> str:
    """processor 注入用：群里最近分享的抖音视频摘要块。"""
    if not _understand_enabled():
        return ""
    now = time.time()
    items = [v for ts, v in _RECENT.get(chat_id, ()) if now - ts <= _RECENT_TTL]
    if not items:
        return ""
    lines = []
    for v in items[-_RECENT_MAX:]:
        line = f"- {v['title']}（作者：{v['owner']}）"
        if v["summary"]:
            line += f"：{v['summary']}"
        lines.append(line)
    return "群里最近分享的抖音视频（你看过它的资料，可以自然聊起）：\n" + "\n".join(lines)
