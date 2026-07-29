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
