"""chat_screenshot 插件：聊天记录气泡式长图截图（迁移自 chat_screenshot_plugin，新架构重写）。

命令：/screenshot [数量]（别名 /截图），默认 20 条、上限 50 条
工具：chat_screenshot(message_count) —— LLM 举证用，渲染后直接发图到当前会话

渲染要点（对齐旧插件样式）：浅色背景、气泡式布局，用户消息居左（绿色气泡）、
bot 消息居右（白色气泡），昵称 + 居中时间戳 + 自动换行，宽 800px。
截图统一存 data/screenshots/shot_<时间戳>.png（目录自动建）。

Pillow 为可选依赖：import 本模块不依赖 PIL，缺失时 probe_available() 返回 False，
加载器会禁用本插件并 WARN，不炸启动。
"""

import time
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from junjun_agent.commands import register_command
from junjun_core.contracts import ReplySegment, ReplySet
from junjun_core.observability import get_logger

logger = get_logger("plugin.chat_screenshot")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "screenshots"

_DEFAULT_COUNT = 20
_MAX_COUNT = 50

# 渲染样式（配色参照旧插件 config.toml）
_WIDTH = 800                 # 图片宽度
_FONT_SIZE = 16              # 正文字号
_BG = (245, 245, 245)        # 背景 #F5F5F5
_USER_BUBBLE = (149, 236, 105)   # 用户气泡 #95EC69（微信绿）
_BOT_BUBBLE = (255, 255, 255)    # bot 气泡 #FFFFFF
_TEXT = (0, 0, 0)            # 文字 #000000
_TS = (153, 153, 153)        # 时间戳 #999999
_RADIUS = 8                  # 气泡圆角
_SPACING = 10                # 消息间距
_PAD = 20                    # 页面边距

_FONT_CANDIDATES = ("C:/Windows/Fonts/msyh.ttc", "msyh.ttc", "arial.ttf")


def probe_available() -> bool:
    """依赖探测：Pillow 可用才启用本插件。"""
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _load_fonts():
    """加载正文字体与小字号字体，取不到系统字体时降级为 PIL 默认字体（不炸）。"""
    from PIL import ImageFont
    for name in _FONT_CANDIDATES:
        try:
            return (ImageFont.truetype(name, _FONT_SIZE),
                    ImageFont.truetype(name, _FONT_SIZE - 4))
        except Exception:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def _text_width(font, s: str) -> float:
    """测文本像素宽度（兼容默认字体）。"""
    try:
        return font.getlength(s)
    except Exception:
        bbox = font.getbbox(s)
        return bbox[2] - bbox[0]


def _wrap_text(text: str, font, max_width: int) -> list:
    """按像素宽度逐字符换行，保留原文换行符。"""
    lines = []
    for para in (text or "").split("\n"):
        cur = ""
        for ch in para:
            if _text_width(font, cur + ch) <= max_width:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines or [""]


def render_image(rows: list, out_path) -> Path:
    """把消息渲染成气泡式聊天记录长图 PNG。

    Args:
        rows: [{nickname, timestamp, text, is_bot}]，is_bot=True 的消息居右（白气泡）
        out_path: 输出 PNG 路径（父目录自动建）
    """
    from PIL import Image, ImageDraw
    font, small_font = _load_fonts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_bubble = int(_WIDTH * 0.6)
    line_h = _FONT_SIZE + 5

    # 第一遍：预排版 + 累计高度
    prepared = []  # (row, lines, bubble_w, bubble_h)
    total_h = _PAD
    for r in rows:
        lines = _wrap_text(r.get("text") or "", font, max_bubble - 20)
        bw = min(max((_text_width(font, ln) for ln in lines), default=0) + 20, max_bubble)
        bh = len(lines) * line_h + 20
        prepared.append((r, lines, bw, bh))
        total_h += 25 + 25 + bh + _SPACING  # 时间戳行 + 昵称行 + 气泡 + 间距

    img = Image.new("RGB", (_WIDTH, total_h + _PAD), _BG)
    draw = ImageDraw.Draw(img)

    y = _PAD
    for r, lines, bw, bh in prepared:
        is_bot = bool(r.get("is_bot"))
        # 时间戳（居中）
        ts = r.get("timestamp") or ""
        if ts:
            draw.text((_WIDTH / 2, y), ts, fill=_TS, font=small_font, anchor="ma")
        y += 25
        # 昵称（与气泡同侧）
        nick = r.get("nickname") or "未知用户"
        if is_bot:
            draw.text((_WIDTH - _PAD, y), nick, fill=_TEXT, font=small_font, anchor="ra")
        else:
            draw.text((_PAD, y), nick, fill=_TEXT, font=small_font)
        y += 25
        # 气泡：bot 居右白底，用户居左绿底
        x1 = _WIDTH - _PAD - bw if is_bot else _PAD
        draw.rounded_rectangle([x1, y, x1 + bw, y + bh], radius=_RADIUS,
                               fill=_BOT_BUBBLE if is_bot else _USER_BUBBLE)
        ty = y + 10
        for line in lines:
            draw.text((x1 + 10, ty), line, fill=_TEXT, font=font)
            ty += line_h
        y += bh + _SPACING

    img.save(out_path, format="PNG")
    return out_path


def _parse_count(args: str) -> int:
    """解析数量参数：非法输入回退默认值，结果夹在 [1, 上限]。"""
    try:
        n = int((args or "").strip())
    except ValueError:
        return _DEFAULT_COUNT
    return max(1, min(_MAX_COUNT, n))


def _fetch_rows(chat_id: str, count: int) -> list:
    """取当前会话最近 count 条消息，按时间从旧到新返回。"""
    from junjun_core.database.models import Messages
    rows = list(
        Messages.select()
        .where(Messages.chat_id == chat_id)
        .order_by(Messages.time.desc())
        .limit(count)
    )
    rows.reverse()
    return rows


def _to_render_row(msg) -> dict:
    """Messages 行 -> 渲染行。"""
    return {
        "nickname": msg.user_nickname or ("我" if msg.is_bot else "未知用户"),
        "timestamp": datetime.fromtimestamp(msg.time).strftime("%Y-%m-%d %H:%M:%S"),
        "text": msg.processed_plain_text or "",
        "is_bot": bool(msg.is_bot),
    }


def _make_screenshot(chat_id: str, count: int):
    """取记录并渲染长图。返回 (图片路径, 条数)；无记录返回 (None, 0)。"""
    rows = _fetch_rows(chat_id, count)
    if not rows:
        return None, 0
    path = DATA_DIR / f"shot_{int(time.time() * 1000)}.png"
    render_image([_to_render_row(r) for r in rows], path)
    logger.info(f"聊天记录截图已生成: {path}（{len(rows)} 条）")
    return path, len(rows)


@register_command("screenshot", aliases=["截图"], plugin="chat_screenshot",
                  description="聊天记录截图。用法：/screenshot [数量]（默认20，上限50）")
async def screenshot_cmd(ctx):
    count = _parse_count(ctx.args)
    path, n = _make_screenshot(ctx.session.chat_id, count)
    if path is None:
        return "没有聊天记录可截图"
    await ctx.send([ReplySegment(type="image", data=str(path))])
    return None


@tool("chat_screenshot")
async def chat_screenshot_tool(message_count: int = 20) -> str:
    """截取当前会话最近的聊天记录，渲染成长图截图并发送到当前会话。
    用户否认说过某些话需要举证、要求"截图聊天记录""翻记录给你看"时使用。

    Args:
        message_count: 截取的消息数量，默认 20，最多 50
    """
    from junjun_core.gateway.router import get_gateway
    from junjun_skills.builtin.memory_skills import current_chat_id

    chat_id = current_chat_id.get()
    count = max(1, min(_MAX_COUNT, int(message_count)))
    path, n = _make_screenshot(chat_id, count)
    if path is None:
        return "没有聊天记录可截图"

    # chat_id 形如 "qq:ID:group|private"，解析出回复目标
    parts = chat_id.split(":")
    platform = parts[0] if parts else "qq"
    target = parts[1] if len(parts) > 1 else ""
    kind = parts[2] if len(parts) > 2 else "private"
    await get_gateway().send_reply(ReplySet(
        platform=platform,
        target_group_id=target if kind == "group" else None,
        target_user_id=target if kind != "group" else None,
        segments=[ReplySegment(type="image", data=str(path))],
        should_reply=True,
    ))
    return f"已生成并发送聊天记录截图（共 {n} 条消息）。"


TOOLS = [chat_screenshot_tool]
