"""Pixiv 官方 AJAX 客户端（共享层）：cookie/请求头/反爬绕过/JSON 抓取。

从 pixiv_novel 插件提取并扩展：
- _fetch_json：body 包装的 AJAX 端点（/ajax/...）
- _fetch_raw ：顶层 JSON 端点（/ranking.php?format=json 无 body 包装）
- 反爬：Cloudflare TLS 指纹检测 -> curl_cffi Chrome 指纹伪装；httpx 兜底
- 403/429/5xx 重试 3 次；4xx 其他确定性错误不重试

配置：插件目录 config.toml（[network] proxy / [features] api_timeout）+
PIXIV_COOKIE env（PHPSESSID）。
"""

import os
import re
import tomllib
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
