"""B站视频内容理解：字幕/弹幕/热评 -> 材料 -> 摘要 -> 上下文注入 + 理解工具。

三个免费数据源（不用下载视频，2-3 秒出结果）：
- AI 字幕：player/v2 -> subtitle_url JSON（讲解/攻略/资讯类视频大多有，信息量最大）
- 弹幕采样：dm/list.so（氛围、名场面、观众反应）
- 热评 TOP：x/v2/reply sort=2（内容梗概、风评、争议点）

字幕命中 -> 材料=简介+字幕；否则降级 简介+弹幕+热评（判断不了细节但能聊话题）。
材料按 bvid 缓存 30 分钟 + in-flight 共享：拦截器预热与 bilibili_summary
工具走的是同一份，群友追问第二次零成本。

会话「最近视频」清单（_RECENT）由 processor 注入上下文——君君像真人一样
「瞄过一眼」群里刚分享的视频，能自然接话。
"""

import asyncio
import re
import time
from collections import deque
from typing import Dict, Optional, Tuple

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("plugin.bilibili.content")

_CACHE_TTL = 1800.0     # 材料缓存 30 分钟
_RECENT_TTL = 1800.0    # 会话「最近视频」有效期 30 分钟
_RECENT_MAX = 3         # 注入上限
_DANMAKU_SAMPLE = 30    # 弹幕均匀采样条数
_REPLY_TOP = 5          # 热评条数

_MATERIAL_CACHE: Dict[str, Tuple[float, dict]] = {}  # bvid -> (ts, material)
_PENDING: Dict[str, asyncio.Task] = {}               # bvid -> in-flight 任务
_RECENT: Dict[str, deque] = {}                       # chat_id -> [(ts, video)]


def _cfg() -> dict:
    try:
        return get_global_config().raw.get("bilibili", {}) or {}
    except Exception:
        return {}


def _understand_enabled() -> bool:
    return bool(_cfg().get("enable_understand", True))


# ---------------------------------------------------------------- 三个数据源

def _pick_subtitle_url(payload: dict) -> str:
    """player/v2 响应 -> 首个字幕 JSON 地址（// 开头补 https）。"""
    subs = (((payload or {}).get("data") or {}).get("subtitle") or {}).get("subtitles") or []
    if not subs:
        return ""
    url = str(subs[0].get("subtitle_url") or "")
    if url.startswith("//"):
        url = "https:" + url
    return url


def _parse_subtitle_body(data: dict, max_chars: int) -> str:
    """字幕 JSON -> 按行拼接的纯文本。"""
    body = (data or {}).get("body") or []
    text = "\n".join(str(b.get("content", "")).strip() for b in body if b.get("content"))
    return text[:max_chars]


def _parse_danmaku(xml: str, sample: int = _DANMAKU_SAMPLE) -> list:
    """弹幕 XML -> 均匀采样（防头部扎堆）。"""
    texts = re.findall(r"<d p=\"[^\"]*\">([^<]{1,60})</d>", xml or "")
    if not texts:
        return []
    step = max(1, len(texts) // sample)
    return texts[::step][:sample]


def _parse_replies(payload: dict, top: int = _REPLY_TOP) -> list:
    """热评响应 -> 评论文本列表。"""
    replies = (((payload or {}).get("data") or {}).get("replies") or [])
    out = []
    for r in replies[:top]:
        msg = str(((r.get("content") or {}).get("message") or "")).strip().replace("\n", " ")
        if msg:
            out.append(msg[:120])
    return out


async def _fetch_subtitle_text(aid: int, cid: int) -> str:
    """AI 字幕全文（按行拼接，截断到配置上限）。没有/失败返回 ""。"""
    from junjun_skills.plugins.bilibili.tools import _fetch_json, _wbi_sign
    max_chars = int(_cfg().get("subtitle_max_chars", 8000))
    try:
        params = await _wbi_sign({"aid": str(aid), "cid": str(cid)})
        payload = await _fetch_json("https://api.bilibili.com/x/player/v2", params=params)
        url = _pick_subtitle_url(payload or {})
        if not url:
            return ""
        data = await _fetch_json(url)
        return _parse_subtitle_body(data or {}, max_chars)
    except Exception as e:
        logger.debug(f"字幕获取失败（降级弹幕热评）: {type(e).__name__}: {e}")
        return ""


async def _fetch_danmaku_sample(cid: int) -> list:
    """弹幕均匀采样。失败返回 []。"""
    from junjun_skills.plugins.bilibili.tools import _HTTP_TIMEOUT, USER_AGENT
    try:
        import httpx
        headers = {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"}
        async with httpx.AsyncClient(headers=headers, timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get("https://api.bilibili.com/x/v1/dm/list.so",
                                    params={"oid": str(cid)})
        return _parse_danmaku(resp.text or "")
    except Exception as e:
        logger.debug(f"弹幕获取失败: {type(e).__name__}: {e}")
        return []


async def _fetch_top_replies(aid: int) -> list:
    """热评 TOP（按热度排序）。失败返回 []。"""
    from junjun_skills.plugins.bilibili.tools import _fetch_json
    try:
        payload = await _fetch_json(
            "https://api.bilibili.com/x/v2/reply",
            params={"type": "1", "oid": str(aid), "sort": "2", "ps": str(_REPLY_TOP)})
        return _parse_replies(payload or {})
    except Exception as e:
        logger.debug(f"热评获取失败: {type(e).__name__}: {e}")
        return []


# ---------------------------------------------------------------- 材料构建（缓存 + 共享）

async def _build_material(bvid: str) -> Optional[dict]:
    from junjun_skills.plugins.bilibili.tools import _fetch_view, _fmt_duration
    info = await _fetch_view(bvid)
    if not info or not info.get("aid"):
        return None
    subtitle, danmaku, replies = await asyncio.gather(
        _fetch_subtitle_text(info["aid"], info["cid"] or 0),
        _fetch_danmaku_sample(info["cid"] or 0),
        _fetch_top_replies(info["aid"]),
    )
    parts = [f"标题：{info['title']}",
             f"UP主：{info.get('owner') or '未知'}",
             f"时长：{_fmt_duration(info.get('duration') or 0)}"]
    if info.get("desc"):
        parts.append(f"简介：{info['desc'][:200]}")
    if subtitle:
        parts.append(f"字幕全文（长视频为节选）：\n{subtitle}")
        source = "字幕"
    else:
        if danmaku:
            parts.append("弹幕采样：\n" + " / ".join(danmaku))
        if replies:
            parts.append("热评：\n" + "\n".join(f"- {r}" for r in replies))
        source = "弹幕热评" if (danmaku or replies) else "简介"
    return {"info": info, "material": "\n".join(parts), "source": source,
            "page_url": f"https://www.bilibili.com/video/{info['bvid']}"}


async def get_material(url: str) -> Optional[dict]:
    """提取视频内容材料（30 分钟缓存 + in-flight 共享）。失败返回 None。"""
    from junjun_skills.plugins.bilibili.tools import extract_bvid
    bvid = await extract_bvid(url)
    if not bvid:
        return None
    hit = _MATERIAL_CACHE.get(bvid)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    task = _PENDING.get(bvid)
    if task is None or task.done():
        task = asyncio.create_task(_build_material(bvid))
        _PENDING[bvid] = task
        task.add_done_callback(lambda _t, b=bvid: _PENDING.pop(b, None))
    try:
        result = await task
    except Exception as e:
        logger.debug(f"视频材料构建失败: {type(e).__name__}: {e}")
        return None
    if result:
        _MATERIAL_CACHE[bvid] = (time.time(), result)
        logger.info(f"B站视频材料: {bvid}（依据{result['source']}，{len(result['material'])} 字）")
    return result


# ---------------------------------------------------------------- 摘要

_SUM_PROMPT = """基于以下 B 站视频的材料，用中文口语两三句话讲清这个视频讲了什么
（{max_chars} 字以内，说重点：主题/结论/亮点/风评）。不要客套，不要「这个视频」之外的前缀。
材料：
{material}"""


async def summarize_material(material: str, *, model=None, max_chars: int = 100) -> str:
    """utils 模型把材料压成口语摘要（上下文注入/工具答复共用）。"""
    if model is None:
        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
    from langchain_core.messages import HumanMessage
    resp = await model.ainvoke([HumanMessage(content=_SUM_PROMPT.format(
        max_chars=max_chars, material=material[:9000]))])
    return str(resp.content).strip()


# ---------------------------------------------------------------- 预热 + 会话最近视频

def prewarm_video(chat_id: str, url: str) -> None:
    """后台「看懂」视频：材料入缓存 + 摘要进会话最近视频清单。fire-and-forget。"""
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
        if not m:
            return
        summary = ""
        try:
            summary = await summarize_material(m["material"])
        except Exception as e:
            logger.debug(f"视频摘要失败（降级简介）: {type(e).__name__}")
        info = m["info"]
        dq = _RECENT.setdefault(chat_id, deque(maxlen=_RECENT_MAX * 2))
        dq.append((time.time(), {
            "title": info["title"], "owner": info.get("owner") or "",
            "summary": summary or (info.get("desc", "") or "")[:80],
            "page_url": m["page_url"],
        }))
    except Exception as e:
        logger.debug(f"视频后台理解失败（忽略）: {type(e).__name__}: {e}")


def render_recent_block(chat_id: str) -> str:
    """processor 注入用：群里最近分享的 B 站视频摘要块。"""
    if not _understand_enabled():
        return ""
    now = time.time()
    items = [v for ts, v in _RECENT.get(chat_id, ()) if now - ts <= _RECENT_TTL]
    if not items:
        return ""
    lines = []
    for v in items[-_RECENT_MAX:]:
        line = f"- 《{v['title']}》（UP主：{v['owner']}）"
        if v["summary"]:
            line += f"：{v['summary']}"
        lines.append(line)
    return "群里最近分享的B站视频（你看过它的字幕资料，可以自然聊起）：\n" + "\n".join(lines)
