"""music 插件：多音源点歌（迁移自 music_player_plugin，提取 API 协议后按新架构重写）。

命令：/music [netease|qq|vip|juhe] 歌名、/choose 序号
拦截器：搜索后 60 秒内直接发纯数字（1-99）快捷选歌，无缓存则放行
工具：play_music（LLM 点歌，直接搜第一个结果发音乐卡片）

API 协议（提取自旧插件）：
- 普通音源 api.vkeys.cn：GET /v2/music/netease|tencent
  搜索参数 word/page/num → {"code":200,"data":[{id,song,singer,album,cover,url,link,interval},...]}
  详情参数 word/choose → {"code":200,"data":{...同上...}}
- VIP www.littleyouzi.com/api/v2/netmusic：
  搜索参数 name/limit → 列表或 {"data":[...]}（字段 name/artist/pic/mid）
  详情参数 mid/level → {"data":{"url"|"mp3",...}}
- 聚合 api.xcvts.cn/api/music/juhe：
  搜索参数 msg/type=json → {"list":[{n,title,singer,app,songid,cover},...]}
  详情参数 msg/n/type=json → {"data":{"code":200,"title","singer","url","cover","link"}}

偏差：旧插件用 PIL 生成歌单图片，本版改为纯文本编号列表。
"""

import json
import math
import os
import time

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_agent.interceptors import register_interceptor
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger
from junjun_skills.builtin.memory_skills import current_chat_id

logger = get_logger("plugin.music")

# ===== 配置（可用环境变量覆盖地址）=====
_API_BASE = os.environ.get("MUSIC_API_BASE", "https://api.vkeys.cn")
_VIP_API_BASE = os.environ.get("MUSIC_VIP_API_BASE", "https://www.littleyouzi.com/api/v2")
_JUHE_API = os.environ.get("MUSIC_JUHE_API", "https://api.xcvts.cn/api/music/juhe")
_TIMEOUT = 10.0
_MAX_RESULTS = 10          # 单次搜索最大返回条数
_CACHE_TTL = 60.0          # 搜索缓存有效期（秒），惰性过期
_RATE_SECONDS = 10.0       # 每会话点歌限流（秒）

_SOURCE_NAMES = {"netease": "网易云音乐", "qq": "QQ音乐", "vip": "网易云VIP", "juhe": "聚合点歌"}
_FALLBACK_ORDER = ("netease", "qq")  # 未指定音源时的降级顺序

# 每会话搜索缓存：chat_id -> {keyword, results, source, timestamp}
_search_cache: dict = {}
# 每会话限流时间戳：chat_id -> 上次点歌时间
_last_use: dict = {}


# ===== HTTP 与数据标准化 =====

async def _get_json(url: str, params: dict):
    """GET JSON，任何失败都返回 None（绝不向上抛异常）。"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
        return resp.json()
    except Exception as e:
        logger.warning(f"音乐 API 请求失败 {url}: {type(e).__name__}: {e}")
        return None


def _norm(item: dict, source: str) -> dict:
    """把各音源返回的歌曲条目标准化成统一结构。"""
    return {
        "source": source,
        "source_name": _SOURCE_NAMES[source],
        "id": str(item.get("id") or item.get("mid") or item.get("songid") or item.get("n") or ""),
        "song": item.get("song") or item.get("name") or item.get("title") or "未知歌曲",
        "singer": item.get("singer") or item.get("artist") or "未知歌手",
        "album": item.get("album") or "未知专辑",
        "cover": item.get("cover") or item.get("pic") or "",
        "url": item.get("url") or item.get("mp3") or "",
        "link": item.get("link") or "",
        "interval": item.get("interval") or item.get("time") or "未知时长",
    }


# ===== 各音源搜索/详情（独立 helper，便于测试替换）=====

async def _search_vkeys(keyword: str, num: int, endpoint: str, source: str):
    """vkeys 网易/QQ 搜索。endpoint: netease|tencent。"""
    data = await _get_json(f"{_API_BASE}/v2/music/{endpoint}",
                           {"word": keyword, "page": 1, "num": num})
    if not isinstance(data, dict) or data.get("code") != 200:
        return None
    items = data.get("data")
    if isinstance(items, dict):
        items = [items]
    if not items:
        return None
    return [_norm(it, source) for it in items[:num]]


async def _search_vip(keyword: str, num: int):
    """littleyouzi 网易 VIP 搜索。"""
    limit = min(max(num, 1), 100)
    data = await _get_json(f"{_VIP_API_BASE}/netmusic", {"name": keyword, "limit": limit})
    if isinstance(data, dict):
        data = data.get("data", data)
    if isinstance(data, dict):
        data = [data]
    if not data:
        return None
    return [_norm(it, "vip") for it in data[:limit]]


async def _search_juhe(keyword: str, num: int):
    """聚合点歌搜索。"""
    data = await _get_json(_JUHE_API, {"msg": keyword, "type": "json"})
    items = data.get("list") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        return None
    return [_norm(it, "juhe") for it in items[:num]]


async def fetch_search(source: str, keyword: str, num: int = _MAX_RESULTS):
    """按音源搜索，返回标准化歌曲列表或 None。"""
    if source == "qq":
        return await _search_vkeys(keyword, num, "tencent", "qq")
    if source == "vip":
        return await _search_vip(keyword, num)
    if source == "juhe":
        return await _search_juhe(keyword, num)
    return await _search_vkeys(keyword, num, "netease", "netease")


async def _detail_vip(keyword: str, index: int):
    """VIP 详情：先搜出列表取第 index 首，再用 mid 换高音质链接。"""
    songs = await _search_vip(keyword, index)
    if not songs or len(songs) < index:
        return None
    selected = songs[index - 1]
    mid = selected.get("id")
    if mid:
        data = await _get_json(f"{_VIP_API_BASE}/netmusic", {"mid": mid, "level": 2})
        if isinstance(data, dict):
            d = data.get("data", data)
            if isinstance(d, dict):
                url = d.get("url") or d.get("mp3")
                if url:
                    selected["url"] = url
    return selected


async def _detail_juhe(keyword: str, index: int):
    """聚合详情：msg + n 取第 index 首。"""
    resp = await _get_json(_JUHE_API, {"msg": keyword, "n": index, "type": "json"})
    d = resp.get("data") if isinstance(resp, dict) else None
    if isinstance(d, dict) and d.get("code") == 200:
        return _norm(d, "juhe")
    return None


async def fetch_detail(source: str, keyword: str, index: int):
    """取第 index 首（1 起）的可播放详情，失败返回 None。"""
    if source == "vip":
        return await _detail_vip(keyword, index)
    if source == "juhe":
        return await _detail_juhe(keyword, index)
    endpoint = "tencent" if source == "qq" else "netease"
    data = await _get_json(f"{_API_BASE}/v2/music/{endpoint}",
                           {"word": keyword, "choose": index})
    if not isinstance(data, dict) or data.get("code") != 200:
        return None
    d = data.get("data")
    if isinstance(d, list):
        d = d[0] if d else None
    return _norm(d, source) if isinstance(d, dict) else None


async def _search_with_fallback(keyword: str, source: str = ""):
    """带降级的搜索：指定源只搜该源，否则按 netease→qq 顺序尝试。返回 (列表, 源) 或 (None, None)。"""
    sources = (source,) if source else _FALLBACK_ORDER
    for s in sources:
        try:
            results = await fetch_search(s, keyword)
        except Exception as e:  # helper 理论不抛，兜底防御
            logger.warning(f"音源 {s} 搜索异常: {type(e).__name__}: {e}")
            results = None
        if results:
            return results, s
    return None, None


# ===== 缓存 / 限流 =====

def _get_cache(chat_id: str):
    """取会话搜索缓存，惰性过期。"""
    cache = _search_cache.get(chat_id)
    if not cache:
        return None
    if time.time() - cache["timestamp"] >= _CACHE_TTL:
        _search_cache.pop(chat_id, None)
        return None
    return cache


def _rate_wait(chat_id: str) -> int:
    """距解除限流还剩几秒，0=可点歌。"""
    last = _last_use.get(chat_id)
    if last is None:
        return 0
    remain = _RATE_SECONDS - (time.time() - last)
    return math.ceil(remain) if remain > 0 else 0


# ===== 发送 =====

def _build_segments(detail: dict) -> list:
    """构建发送段：有音频直链发 music 卡片+信息行；否则降级为纯文本。"""
    info = f"🎵 {detail['song']} - {detail['singer']}（{detail['album']} · {detail['interval']}）"
    audio = detail.get("url") or ""
    if not audio:
        return [ReplySegment(type="text", data=f"拿到歌曲信息啦，但音频直链失效了：\n{info}")]
    card = json.dumps({
        "url": detail.get("link") or audio,   # 卡片跳转链接
        "audio": audio,                        # 音频直链
        "title": detail["song"],
        "content": detail["singer"],
        "image": detail.get("cover") or "",
    }, ensure_ascii=False)
    return [ReplySegment(type="music", data=card), ReplySegment(type="text", data=info)]


async def _choose_and_send(ctx, cache: dict, index: int):
    """选歌并发送，返回出错文本（无错返回 None）。"""
    results = cache["results"]
    if not 1 <= index <= len(results):
        return f"序号超出范围啦，当前列表只有 {len(results)} 首。"
    try:
        detail = await fetch_detail(cache["source"], cache["keyword"], index)
    except Exception as e:
        logger.warning(f"获取歌曲详情异常: {type(e).__name__}: {e}")
        detail = None
    if not detail:
        return "获取歌曲详情失败了，过会儿再试试吧。"
    await ctx.send(_build_segments(detail))
    return None


# ===== 命令与拦截器 =====

@register_command("music", aliases=["点歌"], plugin="music",
                  description="点歌：/music [netease|qq|vip|juhe] 歌名")
async def music_cmd(ctx):
    """搜索歌曲并发编号列表，结果缓存 60 秒供选歌。"""
    chat_id = ctx.session.chat_id
    wait = _rate_wait(chat_id)
    if wait > 0:
        return f"点歌太频繁啦，{wait} 秒后再试吧。"

    parts = ctx.args.strip().split(None, 1)
    source = parts[0].lower() if parts and parts[0].lower() in _SOURCE_NAMES else ""
    if source:
        keyword = parts[1].strip() if len(parts) > 1 else ""
    else:
        keyword = ctx.args.strip()
    if not keyword:
        return ("用法：/music [音源] 歌名\n"
                "可选音源：netease（网易云）、qq（QQ音乐）、vip（网易VIP）、juhe（聚合）\n"
                "不填音源默认按 网易云→QQ 依次尝试")

    _last_use[chat_id] = time.time()
    results, used = await _search_with_fallback(keyword, source)
    if not results:
        return f"没找到《{keyword}》，换个关键词或音源试试？"

    _search_cache[chat_id] = {"keyword": keyword, "results": results,
                              "source": used, "timestamp": time.time()}
    lines = [f"🎵 「{keyword}」搜索结果（{_SOURCE_NAMES[used]}）："]
    lines += [f"{i}. {r['song']} - {r['singer']}" for i, r in enumerate(results, 1)]
    lines.append(f"发 /choose 序号 或直接发数字选歌（{int(_CACHE_TTL)} 秒内有效）")
    return "\n".join(lines)


@register_command("choose", aliases=["选歌"], plugin="music",
                  description="从点歌列表选择：/choose 序号")
async def choose_cmd(ctx):
    """从最近的搜索结果中选第 N 首播放。"""
    cache = _get_cache(ctx.session.chat_id)
    if not cache:
        return "没有进行中的点歌搜索，先用 /music 搜一下歌名吧。"
    arg = ctx.args.strip()
    if not arg.isdigit():
        return "用法：/choose 序号（搜索后也可以直接发数字）"
    return await _choose_and_send(ctx, cache, int(arg))


@register_interceptor(r"^[1-9]\d{0,1}$", plugin="music")
async def quick_choose(ctx) -> bool:
    """搜索后 60 秒内的纯数字消息 → 快捷选歌；无缓存放行给正常决策。"""
    cache = _get_cache(ctx.session.chat_id)
    if not cache:
        return False
    error = await _choose_and_send(ctx, cache, int(ctx.args))
    if error:
        await ctx.reply(error)
    return True


# ===== LLM 工具 =====

@tool
async def play_music(song_name: str, source: str = "") -> str:
    """搜索并播放音乐。用户想听歌、点歌、说"放首歌来听"时使用；用户没给歌名时根据上下文挑一首合适的。

    Args:
        song_name: 歌曲名，可带歌手名提高命中率，如"周杰伦 晴天"
        source: 音源，可选 netease（网易云）/qq（QQ音乐）/vip（网易VIP）/juhe（聚合），留空自动选择
    """
    song_name = song_name.strip()
    if not song_name:
        return "没告诉我要放什么歌呀，先问清楚歌名再试。"
    results, used = await _search_with_fallback(song_name, source.strip().lower())
    if not results:
        return f"没找到《{song_name}》，建议换个关键词或加上歌手名再试。"
    try:
        detail = await fetch_detail(used, song_name, 1)
    except Exception as e:
        logger.warning(f"play_music 获取详情异常: {type(e).__name__}: {e}")
        detail = None
    if not detail:
        return f"搜到《{song_name}》了，但获取播放详情失败，稍后再试。"

    # 从 chat_id（形如 "qq:ID:group|private"）解析发送目标
    chat_id = current_chat_id.get()
    parts = chat_id.split(":")
    platform = parts[0] if parts and parts[0] else "qq"
    target_id = parts[1] if len(parts) > 1 else ""
    kind = parts[2] if len(parts) > 2 else "private"
    from junjun_core.contracts import ReplySet
    from junjun_core.gateway.router import get_gateway
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target_id if kind == "group" else None,
        target_user_id=target_id if kind != "group" else None,
        segments=_build_segments(detail),
        should_reply=True,
    ))
    if detail.get("url"):
        return f"已为你播放《{detail['song']}》- {detail['singer']}（{detail['source_name']}）。"
    return f"找到了《{detail['song']}》- {detail['singer']}，但音频直链失效，只发出了歌曲信息。"


TOOLS = [play_music]
