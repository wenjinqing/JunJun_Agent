"""回复分割器：长回复拆多条气泡（拟人化核心）。

对齐原 [response_splitter] 语义：
- 按标点（，,。;！？!? 换行）分割，句末标点必切，逗号类概率性合并
- max_sentence_num 上限，超出按 enable_overflow_return_all 合并整发
- max_chars_per_message 单条硬上限
- 颜文字保护（可关）
- 去除包裹中文的括号内容（舞台说明）
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


def _split_sentences(text: str, rng) -> List[str]:
    """按标点切句：句末标点必切，逗号/分号概率性合并（短文本更倾向合并）。"""
    parts = _SPLIT_RE.split(text)
    merge_p = 0.6 if len(text) < 60 else (0.35 if len(text) < 150 else 0.15)

    sentences: List[str] = []
    buf = ""
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if seg:
            buf += seg
        if not buf:
            continue
        if sep in _HARD_STOPS or not sep:
            sentences.append(buf)
            buf = ""
        elif rng.random() > merge_p:
            # 逗号处切开，丢掉句中分隔符（气泡感）
            sentences.append(buf)
            buf = ""
        else:
            # 合并：保留逗号连接
            buf += "，"
    if buf:
        sentences.append(buf.rstrip("，"))
    return [s for s in (x.strip() for x in sentences) if s]


def _hard_wrap(sentence: str, max_chars: int) -> List[str]:
    """超长强拆：优先在换行/标点处断，找不到才按字数硬切。"""
    out: List[str] = []
    rest = sentence
    break_chars = "\n。！!？?；;，, "
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max((window.rfind(c) for c in break_chars), default=-1)
        if cut < max_chars // 3:  # 断点太靠前等于没断，硬切
            cut = max_chars - 1
        out.append(rest[:cut + 1].strip())
        rest = rest[cut + 1:].strip()
    if rest:
        out.append(rest)
    return [p for p in out if p]


def split_response(
    text: str,
    *,
    enable: bool = True,
    max_sentence_num: int = 5,
    max_chars_per_message: int = 120,
    enable_kaomoji_protection: bool = False,
    enable_overflow_return_all: bool = True,
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

    sentences = _split_sentences(text, rand or random)
    if not sentences:
        return []

    if len(sentences) > max_sentence_num:
        if enable_overflow_return_all:
            sentences = [text]  # 一次性整发（原语义：不轰炸群，整段发出）
        else:
            sentences = sentences[:max_sentence_num]

    out: List[str] = []
    for s in sentences:
        if len(s) > max_chars_per_message:
            out.extend(_hard_wrap(s, max_chars_per_message))
        else:
            out.append(s)
    return out


def typing_delay(text: str, *, base: float = 0.4, per_char: float = 0.08, cap: float = 3.0,
                 rand: Optional[random.Random] = None) -> float:
    """按字数模拟打字延迟（秒），带 ±20% 抖动。"""
    rng = rand or random
    d = min(base + len(text) * per_char, cap)
    return d * rng.uniform(0.8, 1.2)
