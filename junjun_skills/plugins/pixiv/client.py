"""Pixiv 官方 AJAX 客户端（共享层）：cookie/请求头/反爬绕过/JSON 抓取。

从 pixiv_novel 插件提取并扩展：
- _fetch_json：body 包装的 AJAX 端点（/ajax/...）
- _fetch_raw ：顶层 JSON 端点（/ranking.php?format=json 无 body 包装）
- 反爬：Cloudflare TLS 指纹检测 -> curl_cffi Chrome 指纹伪装；httpx 兜底
- 403/429/5xx 重试 3 次；4xx 其他确定性错误不重试
- fetch_image_b64：图片本侧代下转 base64://——NapCat 所在网络拉不到
  图床（i.pixiv.re/i.pximg.net 均被墙，2026-08-03 实锤 Connect Timeout），
  发图必须把字节交给 NapCat 而不是 URL

配置：插件目录 config.toml（[network] proxy / [features] api_timeout）+
PIXIV_COOKIE env（PHPSESSID）。
"""

import asyncio
import base64
import os
import re
import tomllib
import urllib.parse
from pathlib import Path

import httpx

from junjun_core.observability import get_logger

logger = get_logger("plugin.pixiv")

_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

BASE_URL = "https://www.pixiv.net"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


# ------------------------------------------------------------------ 配置

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        logger.warning("pixiv/config.toml 不存在，使用默认值")
        return {}
    except Exception as e:
        logger.warning(f"pixiv/config.toml 解析失败: {type(e).__name__}: {e}")
        return {}


def _cfg_section(name: str) -> dict:
    return _load_config().get(name, {}) or {}


def _proxy() -> str:
    return str(_cfg_section("network").get("proxy", "") or "").strip()


def _api_timeout() -> float:
    try:
        return float(_cfg_section("features").get("api_timeout", 30))
    except (TypeError, ValueError):
        return 30.0


def _cookie() -> str:
    """从 env 读 Pixiv Cookie；接受 PHPSESSID=xxx / 整串 cookie / 裸 session 值。"""
    raw = os.environ.get("PIXIV_COOKIE", "").strip()
    if raw and "=" not in raw:
        raw = "PHPSESSID=" + raw
    return raw


# ------------------------------------------------------------------ 请求层

def _headers(referer: str = "") -> dict:
    """构造 Pixiv AJAX 请求头（UA + Referer + Cookie + x-user-id）。"""
    headers = {
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ja-JP,ja;q=0.9,zh-CN;q=0.8,zh;q=0.7,en;q=0.6",
        "Referer": referer or (BASE_URL + "/"),
    }
    cookie = _cookie()
    if cookie:
        headers["Cookie"] = cookie
        m = re.search(r"PHPSESSID=(\d+)_", cookie)
        if m:
            headers["x-user-id"] = m.group(1)
    return headers


async def _get_json(url: str, referer: str = "") -> dict:
    """请求 Pixiv JSON（原样返回整个响应 JSON）。任何失败返回 {"error": ...}。"""
    proxy = _proxy() or None
    timeout = _api_timeout()
    last_status = 0
    for attempt in (1, 2, 3):
        try:
            try:
                from curl_cffi.requests import AsyncSession
                async with AsyncSession(impersonate="chrome", proxy=proxy) as s:
                    resp = await s.get(url, headers=_headers(referer), timeout=timeout)
            except ImportError:
                async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                    resp = await client.get(url, headers=_headers(referer))
            if resp.status_code == 200:
                return resp.json()
            last_status = resp.status_code
            if resp.status_code not in (403, 429, 500, 502, 503):
                break  # 4xx 确定性错误不重试
        except Exception as e:
            logger.warning(f"Pixiv 请求异常（第 {attempt} 次）: {type(e).__name__}: {e}")
            last_status = 0
        if attempt < 3:
            import asyncio
            await asyncio.sleep(1.0 * attempt)
    tip = "（Cloudflare 反爬挑战页，检查代理或稍后再试）" if last_status == 403 else ""
    logger.warning(f"Pixiv 请求失败 HTTP {last_status}{tip}: {url}")
    return {"error": f"HTTP {last_status}" if last_status else "网络请求失败"}


async def _fetch_json(url: str, referer: str = "") -> dict:
    """body 包装的 AJAX 端点（/ajax/...）：解包返回 body；body 是 list 时原样返回。"""
    data = await _get_json(url, referer)
    if not isinstance(data, dict):
        return {"error": "响应格式异常"}
    err = data.get("error")
    if err:
        # 传输层失败是 _get_json 的短路 {"error": "HTTP 403"}（str 且无 body）；
        # Pixiv 应用层失败是 {"error": true, "message": "..."}（bool）
        if isinstance(err, str) and "body" not in data:
            return data
        return {"error": str(data.get("message") or "Pixiv 返回错误")}
    return data.get("body", {}) or {}


async def _fetch_raw(url: str, referer: str = "") -> dict:
    """顶层 JSON 端点（/ranking.php?format=json 无 body 包装）：原样返回。"""
    data = await _get_json(url, referer)
    if not isinstance(data, dict):
        return {"error": "响应格式异常"}
    err = data.get("error")
    if err and not isinstance(err, str):
        return {"error": str(data.get("message") or "Pixiv 返回错误")}
    return data


def pximg_proxy(url: str) -> str:
    """i.pximg.net 需要 Referer 才能下载，NapCat 拉图不带 Referer——
    改走公共代理 i.pixiv.re（lolicon 同款技巧）。"""
    return (url or "").replace("i.pximg.net", "i.pixiv.re")


# ------------------------------------------------------------------ 内容安全（元数据层）

# R18 tag 黑名单（2026-08-03 用户实锤：xRestrict=0 + mode=safe 也会漏擦边/R18）
_R18_TAGS = frozenset({
    "r-18", "r18", "r-18g", "r18g", "nsfw", "エロ", "えろ",
    "裸体", "ヌード", "全裸", "セックス", "性交", "中出し", "陵辱",
})


def item_tags(item: dict) -> list:
    """统一提取 tag 字符串列表：搜索条目是 list[str]，详情是 {"tags": [{"tag": ...}]}。"""
    raw = item.get("tags") or []
    if isinstance(raw, dict):
        raw = raw.get("tags") or []
    return [str(t.get("tag") or "") if isinstance(t, dict) else str(t) for t in raw]


def has_r18_tag(item: dict) -> bool:
    return any(t.strip().lower() in _R18_TAGS for t in item_tags(item))


def sl_value(item: dict) -> int:
    """搜索条目的 sl（露骨分级）：0/2 正常，4 擦边，6 露骨。拿不到按 0。"""
    try:
        return int(item.get("sl") or 0)
    except (TypeError, ValueError):
        return 0


def is_safe_item(item: dict, group: bool) -> bool:
    """R18 综合过滤（元数据层，0 额外请求）：
    - 通用：xRestrict==0 + R18 tag 黑名单
    - 群聊加码：sl<=2（sl 4/6 的擦边/露骨会带着 xRestrict=0 漏过 mode=safe）
    """
    try:
        if int(item.get("xRestrict") or 0) >= 1:
            return False
    except (TypeError, ValueError):
        pass
    if has_r18_tag(item):
        return False
    if group and sl_value(item) > 2:
        return False
    return True


def _min_bookmarks() -> int:
    """随机发图的收藏数门槛（质量兜底，config [features] min_bookmarks）。"""
    try:
        return int(_cfg_section("features").get("min_bookmarks", 300))
    except (TypeError, ValueError):
        return 300


# ------------------------------------------------------------------ 搜索（免会员质量方案）

def quality_tiers() -> list:
    """收藏分层 tag 序列：配置层 -> 逐级放宽 -> 裸关键词（去重保序）。

    2026-08-03 实锤：popular_d（人気順）是 Premium 限定，非会员账号
    【静默降级为最新优先】——出来的全是几小时前的新投稿（丑+低收藏）。
    免会员替代：关键词 + 「1000users入り」收藏分层 tag（pixiv 自动给
    高收藏作品打的标签）搜索，效果接近人気順。
    """
    tier = str(_cfg_section("features").get("quality_tier", "1000users入り") or "")
    tiers = [tier, "500users入り", "100users入り", ""] if tier else [""]
    return list(dict.fromkeys(tiers))


async def search_artworks(query: str, page: int, ratio: str = "") -> list:
    """/ajax/search/artworks 封装（date_d + safe + s_tag），返回原始条目列表。

    注意不要用 popular_d：Premium 限定，非会员静默降级 date_d（2026-08-03 实锤）。
    """
    enc = urllib.parse.quote(query)
    url = (BASE_URL + f"/ajax/search/artworks/{enc}?word={enc}"
           f"&order=date_d&mode=safe&p={page}&s_mode=s_tag"
           + (f"&ratio={ratio}" if ratio else ""))
    body = await _fetch_json(url, BASE_URL + "/tags/")
    if body.get("error"):
        return []
    return (body.get("illustManga") or {}).get("data") or []


# ------------------------------------------------------------------ 图片代下

async def fetch_image_b64(url: str) -> str:
    """下载图片 -> "base64://..." 串；失败返回 ""。

    2026-08-03 实锤：NapCat 直连图床 Connect Timeout（被墙+无代理），
    发图必须本侧（有代理）下载后把字节交给 NapCat，而不是给 URL。
    """
    if not url:
        return ""
    proxy = _proxy() or None
    headers = {"User-Agent": _UA, "Referer": BASE_URL + "/"}
    try:
        try:
            from curl_cffi.requests import AsyncSession
            async with AsyncSession(impersonate="chrome", proxy=proxy) as s:
                resp = await s.get(url, headers=headers, timeout=_api_timeout())
        except ImportError:
            async with httpx.AsyncClient(timeout=_api_timeout(), proxy=proxy) as client:
                resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            logger.warning(f"图片下载失败 HTTP {resp.status_code}: {url}")
            return ""
        return "base64://" + base64.b64encode(resp.content).decode()
    except Exception as e:
        logger.warning(f"图片下载异常: {type(e).__name__}: {e} ({url})")
        return ""


async def images_to_b64(urls: list) -> list:
    """并发代下 -> base64:// 列表（保序，失败的跳过）。"""
    results = await asyncio.gather(*(fetch_image_b64(u) for u in urls))
    return [r for r in results if r]
