"""回复分割器：长回复拆多条气泡（拟人化核心）。

对齐原 [response_splitter] 语义：
- 按标点（，,。;！？!? 换行 ——）分割，句末标点必切，逗号类概率性合并
- max_sentence_num 上限，超出按 enable_overflow_return_all 合并整发
- max_chars_per_message 单条硬上限
- 颜文字保护（可关）
- 去除包裹中文的括号内容（舞台说明）

2026-07-29 调整（用户反馈）：
- 切分点落在逗号/句号（，,。.）时，发出的气泡**不带**该标点
  （连续气泡自带停顿感，句尾逗号/句号显得机器人）；！？；、—— 等保留
- 省略号 ... 与数字小数点不误伤（只丢句尾单个 .）
- overflow 整发时保留原文标点（一整段完整的话）
"""

import random
import re
from typing import List, Optional

# 常见颜文字模式（简化版，覆盖原 protect_kaomoji 主要形态）
_KAOMOJI_RE = re.compile(
    r"[（(][^（()）]{0,12}[・ω´｀∀︿⌒▽°□゜ノシo〇^~\-_=+*'\"`;:,.<>/\\|!?？！]{2,}[^（()）]{0,12}[)）]"
)
# 包裹中文的括号内容（舞台说明如「（摸摸头）」）
_CN_PAREN_RE = re.compile(r"[(\[（](?=[^)\]）]*[一-鿿])[^)\]）]*[)\]）]")
_SPLIT_RE = re.compile(r"([，,。;；！!？?——\n])")
_HARD_STOPS = frozenset("。！!？?——\n")
_STRIP_TAIL_CHARS = "，,。"  # 切分点丢弃的标点（英文 . 单独处理，防误伤 ...）


def _protect_kaomoji(text: str):
    mapping = {}

    def repl(m):
        key = f"\x00K{len(mapping)}\x00"
        mapping[key] = m.group(0)
        return key

    return _KAOMOJI_RE.sub(repl, text), mapping


def _restore_kaomoji(text: str, mapping: dict) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def _strip_stage_directions(text: str, protect_kaomoji: bool) -> str:
    """去掉包裹中文的括号内容；开颜文字保护时先摘出颜文字。"""
    if protect_kaomoji:
        protected, mapping = _protect_kaomoji(text)
        cleaned = _CN_PAREN_RE.sub("", protected)
        return _restore_kaomoji(cleaned, mapping)
    return _CN_PAREN_RE.sub("", text)


def _strip_tail_punct(s: str, strip_punct: bool) -> str:
    """丢句尾的逗号/句号（切分点标点）；省略号 ... 与小数点不误伤。"""
    if not strip_punct:
        return s
    while s and s[-1] in _STRIP_TAIL_CHARS:
        s = s[:-1]
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1]
    return s


def _split_sentences(text: str, rng, strip_punct: bool = True) -> List[str]:
    """按标点切句：句末标点必切，逗号/分号概率性合并（短文本更倾向合并）。

    切分点的逗号/句号不随气泡发出（strip_punct），其余标点保留。
    """
    parts = _SPLIT_RE.split(text)
    merge_p = 0.8 if len(text) < 60 else (0.5 if len(text) < 150 else 0.25)

    sentences: List[str] = []
    buf = ""
    for i in range(0, len(parts), 2):
        seg = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if seg:
            buf += seg
        if not buf:
            continue
        if sep in _HARD_STOPS or not sep:
            # 硬切：句末标点随句（句号会被 strip，！？保留）
            sentences.append(_strip_tail_punct(buf + (sep if sep else ""), strip_punct))
            buf = ""
        elif rng.random() > merge_p:
            # 逗号处切开：逗号不随气泡发出
            sentences.append(_strip_tail_punct(buf + sep, strip_punct))
            buf = ""
        else:
            # 合并：逗号在句中，保留不动
            buf += sep if sep else "，"
    if buf:
        sentences.append(_strip_tail_punct(buf, strip_punct))
    return [s for s in (x.strip() for x in sentences) if s]


def _hard_wrap(sentence: str, max_chars: int, strip_punct: bool = True) -> List[str]:
    """超长强拆：优先在换行/标点处断，找不到才按字数硬切。"""
    out: List[str] = []
    rest = sentence
    break_chars = "\n。！!？?；;，, "
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max((window.rfind(c) for c in break_chars), default=-1)
        if cut < max_chars // 3:  # 断点太靠前等于没断，硬切
            cut = max_chars - 1
        out.append(_strip_tail_punct(rest[:cut + 1].strip(), strip_punct))
        rest = rest[cut + 1:].strip()
    if rest:
        out.append(_strip_tail_punct(rest, strip_punct))
    return [p for p in out if p]


def split_response(
    text: str,
    *,
    enable: bool = True,
    max_sentence_num: int = 5,
    max_chars_per_message: int = 120,
    enable_kaomoji_protection: bool = False,
    enable_overflow_return_all: bool = True,
    strip_split_punct: bool = True,
    rand: Optional[random.Random] = None,
) -> List[str]:
    """拆分回复为多条消息。返回非空字符串列表（输入为空时返回空列表）。"""
    text = (text or "").strip()
    if not text:
        return []
    text = _strip_stage_directions(text, enable_kaomoji_protection)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if not text:
        return []
    if not enable:
        return [text]

    sentences = _split_sentences(text, rand or random, strip_punct=strip_split_punct)
    if not sentences:
        return []

    if len(sentences) > max_sentence_num:
        if enable_overflow_return_all:
            sentences = [text]  # 一次性整发（保留原文标点：一整段完整的话）
        else:
            sentences = sentences[:max_sentence_num]

    out: List[str] = []
    for s in sentences:
        if len(s) > max_chars_per_message:
            out.extend(_hard_wrap(s, max_chars_per_message, strip_punct=strip_split_punct))
        else:
            out.append(s)
    return out


def typing_delay(text: str, *, base: float = 0.4, per_char: float = 0.08, cap: float = 3.0,
                 rand: Optional[random.Random] = None) -> float:
    """按字数模拟打字延迟（秒），带 ±20% 抖动。"""
    rng = rand or random
    d = min(base + len(text) * per_char, cap)
    return d * rng.uniform(0.8, 1.2)
