"""image_viewer 插件：随机美图（迁移自 image_viewer_plugin，新架构重写）。

命令（raw 关键词）：看看腿/康康腿/看腿、看看JK、看看白丝、看看黑丝
  也支持 /kankan [tui|jk|baisi|heisi]
API：
  腿   www.onexiaolaji.cn/RandomPicture/api/?class=1|2 （直接 302 到图）
  JK   v2.xxapi.cn/api/jk     -> JSON data = 图直链
  白丝 v2.xxapi.cn/api/baisi  -> 同上
  黑丝 v2.xxapi.cn/api/heisi  -> 同上
"""

import random

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.image_viewer")

_TIMEOUT = 15.0
_TUI_API = "https://www.onexiaolaji.cn/RandomPicture/api/"
_XXAPI = {"jk": "https://v2.xxapi.cn/api/jk",
          "baisi": "https://v2.xxapi.cn/api/baisi",
          "heisi": "https://v2.xxapi.cn/api/heisi"}


async def _fetch_tui_url() -> str | None:
    """腿图：API 直接返回图片流，取最终 URL 作为 image 段（NapCat 会自己下载）。"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(_TUI_API, params={"class": random.choice(["1", "2"])})
            if resp.status_code == 200:
                return str(resp.url)
    except Exception as e:
        logger.warning(f"腿图请求失败: {type(e).__name__}: {e}")
    return None


async def _fetch_xxapi(kind: str) -> str | None:
    """JK/白丝/黑丝：JSON data 字段是图直链。"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_XXAPI[kind])
            data = resp.json()
        url = data.get("data")
        return url if isinstance(url, str) and url.startswith("http") else None
    except Exception as e:
        logger.warning(f"{kind} 图请求失败: {type(e).__name__}: {e}")
    return None


async def _send_pic(ctx, url: str | None) -> str | None:
    if not url:
        return "图片获取失败了，稍后再试试吧。"
    await ctx.send([ReplySegment(type="text", data="看吧！涩批！"),
                    ReplySegment(type="image", data=url)])
    return None


@register_command("看看腿", aliases=["康康腿", "看腿"], raw=True, plugin="image_viewer",
                  description="随机腿图")
async def tui_cmd(ctx):
    return await _send_pic(ctx, await _fetch_tui_url())


@register_command("看看JK", aliases=["看看jk"], raw=True, plugin="image_viewer",
                  description="随机 JK 图")
async def jk_cmd(ctx):
    return await _send_pic(ctx, await _fetch_xxapi("jk"))


@register_command("看看白丝", raw=True, plugin="image_viewer", description="随机白丝图")
async def baisi_cmd(ctx):
    return await _send_pic(ctx, await _fetch_xxapi("baisi"))


@register_command("看看黑丝", raw=True, plugin="image_viewer", description="随机黑丝图")
async def heisi_cmd(ctx):
    return await _send_pic(ctx, await _fetch_xxapi("heisi"))


@register_command("kankan", plugin="image_viewer",
                  description="/kankan [tui|jk|baisi|heisi] 随机美图")
async def kankan_cmd(ctx):
    kind = (ctx.args or "tui").strip().lower() or "tui"
    if kind == "tui":
        return await _send_pic(ctx, await _fetch_tui_url())
    if kind in _XXAPI:
        return await _send_pic(ctx, await _fetch_xxapi(kind))
    return "用法：/kankan [tui|jk|baisi|heisi]"


TOOLS = []
