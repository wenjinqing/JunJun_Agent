r"""Markdown 残留清理：LLM 爱吐 **粗体**、# 标题等标记，QQ 纯文本原样露出很出戏。

规则（2026-07-29 用户反馈「消息经常出现 ** 放大字体描述」）：
- **粗体** / __粗体__ -> 保留文字去标记
- *斜体* / _斜体_ -> 保留文字去标记（仅成对且内容含字母/中文，
  防误伤 3*5*6 这类算式）
- 行首 # ~ ###### 标题号 -> 去掉
- `行内代码` -> 保留文字去反引号
- 先摘出颜文字（(*/ω\*) 内含 *，不保护会被误删成面瘫）
- 不成对的单个 * / _ / # 行中、列表符等一律不动
"""

import re

from junjun_agent.postprocess.splitter import _protect_kaomoji, _restore_kaomoji

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.M)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_HAS_WORD_RE = re.compile(r"[A-Za-z一-鿿]")


def _strip_if_words(m: re.Match) -> str:
    """成对标记才去：内容不含字母/中文（纯数字/符号，多为算式）则原样保留。"""
    inner = m.group(1) if m.group(1) is not None else m.group(2)
    if not _HAS_WORD_RE.search(inner):
        return m.group(0)
    return inner


def clean_markdown(text: str) -> str:
    """剥 markdown 标记，保留文字本体。非 markdown 文本原样返回。"""
    if not text or not re.search(r"[*_#`]", text):
        return text
    protected, mapping = _protect_kaomoji(text)
    protected = _BOLD_RE.sub(_strip_if_words, protected)
    protected = _ITALIC_RE.sub(_strip_if_words, protected)
    protected = _HEADING_RE.sub("", protected)
    protected = _INLINE_CODE_RE.sub(r"\1", protected)
    return _restore_kaomoji(protected, mapping)


# ---------- 语气词 / emoji 清理（2026-08-17 用户拍板：回复不要语气词、不要 emoji） ----------
# prompt 劝告压不过弱模型（reply_style 早写了「不用 emoji」照吐）——结构层兜底。
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF"      # 表情/符号大图区（笑脸、手势、动物、物品、国旗）
    "\U00002600-\U000027BF"       # 杂项符号+装饰符（☀♥✨✅等）
    "\U00002B00-\U00002BFF"       # 星标补充（⭐⬜等）
    "\uFE0F\u200D\u20E3"      # 变体选择符 / ZWJ / 键帽组合符
    "]+",
    flags=re.UNICODE,
)

# 句末语气词白名单：只收「纯语气、构词风险低」的——
# 不收：吗/么/吧（疑问建议语义）、的/了（结构助词）、
#       哈（嘻哈/哈哈哈笑声本体）、呐（唢呐）、哟（哎哟）、哇（惊叹本体）。
_TONE_PARTICLES = "啦呢哦呀嘛咯喽呗喔噢啊"
# 句末位判定允许的后随字符：终止标点/空白/emoji（emoji 留在行尾不挡语气词判定，
# 否则「好呀。🥳」在 emoji 开关关闭时语气词剥不掉，两个开关语义就耦合了）
_TAIL_CTX = (r"。！？!?…\s"
             r"\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
             r"\uFE0F\u200D\u20E3")
_TILDE_TAIL_RE = re.compile(rf"[~～]+(?=[{_TAIL_CTX}]*$)")
_PARTICLE_TAIL_RE = re.compile(
    rf"[{_TONE_PARTICLES}]+(?=[{_TAIL_CTX}]*$)")
_HAS_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9一-鿿]")


def _clean_tone_line(line: str, strip_particles: bool) -> str:
    """单行处理：先剥句尾波浪号，再剥句末语气词（句号/问号/感叹号前或行尾）。

    只剥「行尾/句末」位置：句中逗号前的「他呢」「这个嘛」是话题标记，
    剥了语句不通（宁漏勿错杀）。剥完行里不剩字的（「啊！」->「！」）
    回滚——空感叹号比语气词更怪。
    """
    original = line
    line = _TILDE_TAIL_RE.sub("", line)
    if strip_particles:
        line = _PARTICLE_TAIL_RE.sub("", line)
    if not _HAS_WORD_CHAR_RE.search(line) and _HAS_WORD_CHAR_RE.search(original):
        return original
    return line


def clean_tone(text: str, *, strip_emoji: bool = True,
               strip_particles: bool = True) -> str:
    """回复文本去 emoji + 去句末语气词（含句尾波浪号）。逐行处理。"""
    if not text:
        return text
    if strip_emoji:
        text = _EMOJI_RE.sub("", text)
    if strip_particles:
        text = "\n".join(_clean_tone_line(ln, True) for ln in text.split("\n"))
    else:
        text = "\n".join(_clean_tone_line(ln, False) for ln in text.split("\n"))
    return text
