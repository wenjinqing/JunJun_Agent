"""image_viewer 插件：随机美图（迁移自 image_viewer_plugin，新架构重写）。

命令（raw 关键词）：看看腿/康康腿/看腿、看看胖次、看看JK、看看白丝、看看黑丝
  也支持 /kankan [tui|pangci|jk|baisi|heisi]
API：
  腿/胖次 api.lolicon.app/setu/v2?tag=腿|胖次&r18=0
          （Pixiv 聚合涩图 API，JSON data[0].urls.original = 图直链；
          2026-08-18 替换死掉的 www.onexiaolaji.cn——连接超时确认报废）
  JK     v2.xxapi.cn/api/jk     -> JSON data = 图直链
  白丝   v2.xxapi.cn/api/baisi  -> 同上
  黑丝   v2.xxapi.cn/api/heisi  -> 同上
"""

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment
from junjun_core.observability import get_logger

logger = get_logger("plugin.image_viewer")

_TIMEOUT = 15.0
_LOLICON_API = "https://api.lolicon.app/setu/v2"
_XXAPI = {"jk": "https://v2.xxapi.cn/api/jk",
          "baisi": "https://v2.xxapi.cn/api/baisi",
          "heisi": "https://v2.xxapi.cn/api/heisi"}


async def _fetch_lolicon(tag: str) -> str | None:
    """腿/胖次：Lolicon setu v2 按 tag 抽图，r18=0 只出全年龄（群聊安全）。

    抽中先 HEAD 验活再返回：索引里约 1/5 是已删作品（2026-08-18 实测 6 抽
    2 张 404），NapCat 按 URL 下载会「下载文件失败: Not Found」整消息炸掉，
    404 重抽至多 3 次。
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for attempt in range(3):
                resp = await client.get(_LOLICON_API,
                                        params={"tag": tag, "r18": "0", "num": "1"})
                data = resp.json().get("data") or []
                if not data:
                    return None
                url = data[0].get("urls", {}).get("original")
                if not (isinstance(url, str) and url.startswith("http")):
                    return None
                try:
                    head = await client.head(url)
                    if head.status_code == 200:
                        return url
                    logger.info(f"lolicon[{tag}] 抽中已删作品（{head.status_code}），重抽 {attempt + 1}/3")
                except Exception:
                    return url   # 验活请求本身失败（网络抖动）时宁可放行，不误杀
    except Exception as e:
        logger.warning(f"lolicon[{tag}] 图请求失败: {type(e).__name__}: {e}")
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
    return await _send_pic(ctx, await _fetch_lolicon("腿"))


@register_command("看看胖次", aliases=["康康胖次", "看胖次"], raw=True, plugin="image_viewer",
                  description="随机胖次图")
async def pantsu_cmd(ctx):
    return await _send_pic(ctx, await _fetch_lolicon("胖次"))


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
                  description="/kankan [tui|pangci|jk|baisi|heisi] 随机美图")
async def kankan_cmd(ctx):
    kind = (ctx.args or "tui").strip().lower() or "tui"
    if kind == "tui":
        return await _send_pic(ctx, await _fetch_lolicon("腿"))
    if kind == "pangci":
        return await _send_pic(ctx, await _fetch_lolicon("胖次"))
    if kind in _XXAPI:
        return await _send_pic(ctx, await _fetch_xxapi(kind))
    return "用法：/kankan [tui|pangci|jk|baisi|heisi]"


TOOLS = []
