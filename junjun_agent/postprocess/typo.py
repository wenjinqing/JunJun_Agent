"""中文错别字生成器（拟人化）。

对齐原 chat/utils/typo_generator.py 语义（简化实现）：
- 单字替换：同/近音字替换（pypinyin 同音 + jieba 词频过滤生僻字）
- 声调错误：概率忽略声调找同音字
- 整词替换：jieba 分词后按拼音找同音词
- 白名单跳过：URL / 英文串 / 数字 / @提及
"""

import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

import jieba
from pypinyin import Style, lazy_pinyin, pinyin

# 不动的片段：URL、英文数字串、@xxx、[picid:xxx] 类标记
_PROTECT_RE = re.compile(r"(https?://\S+|\[[a-z]+:[^\]]+\]|@\S+|[A-Za-z0-9_.:/\\-]{2,})")
_CN_CHAR_RE = re.compile(r"[一-鿿]")


@lru_cache(maxsize=1)
def _char_pool() -> Dict[str, List[str]]:
    """拼音(带调) -> 常用字列表。从 jieba 词典高频字构建，惰性一次。"""
    freq: Dict[str, int] = defaultdict(int)
    import os
    dict_path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split(" ")
            if len(parts) >= 2:
                word, cnt = parts[0], int(parts[1])
                for ch in word:
                    if _CN_CHAR_RE.match(ch):
                        freq[ch] += cnt
    pool: Dict[str, List[str]] = defaultdict(list)
    # 只收频率靠前的字（约 3500 常用字规模），避免生僻字
    common = sorted(freq.items(), key=lambda x: -x[1])[:3500]
    for ch, _ in common:
        py = pinyin(ch, style=Style.TONE3, errors="ignore")
        if py and py[0]:
            pool[py[0][0]].append(ch)
    return dict(pool)


def _same_sound_char(ch: str, tone_error: bool, rng: random.Random) -> Optional[str]:
    py = pinyin(ch, style=Style.TONE3, errors="ignore")
    if not py or not py[0]:
        return None
    key = py[0][0]
    pool = _char_pool()
    candidates = list(pool.get(key, []))
    if tone_error:
        # 忽略声调：合并同拼音不同调
        base = key.rstrip("12345")
        for k, chars in pool.items():
            if k.rstrip("12345") == base and k != key:
                candidates.extend(chars)
    candidates = [c for c in candidates if c != ch]
    return rng.choice(candidates) if candidates else None


class ChineseTypoGenerator:
    def __init__(self, error_rate: float = 0.01, min_freq: int = 9,
                 tone_error_rate: float = 0.1, word_replace_rate: float = 0.006):
        self.error_rate = error_rate
        self.tone_error_rate = tone_error_rate
        self.word_replace_rate = word_replace_rate
        self.min_freq = min_freq

    def create_typo_sentence(self, text: str, rand: Optional[random.Random] = None) -> str:
        """生成带错别字的句子。保护 URL/英文/数字/@。"""
        rng = rand or random
        out_parts: List[str] = []
        pos = 0
        for m in _PROTECT_RE.finditer(text):
            out_parts.append(self._typo_segment(text[pos:m.start()], rng))
            out_parts.append(m.group(0))  # 保护段原样
            pos = m.end()
        out_parts.append(self._typo_segment(text[pos:], rng))
        return "".join(out_parts)

    def _typo_segment(self, seg: str, rng: random.Random) -> str:
        if not seg:
            return seg
        # 整词替换（低概率）
        words = list(jieba.cut(seg))
        for i, w in enumerate(words):
            if len(w) >= 2 and _CN_CHAR_RE.search(w) and rng.random() < self.word_replace_rate:
                replaced = self._same_sound_word(w, rng)
                if replaced:
                    words[i] = replaced
        seg = "".join(words)
        # 单字替换
        chars = list(seg)
        for i, ch in enumerate(chars):
            if _CN_CHAR_RE.match(ch) and rng.random() < self.error_rate:
                tone_err = rng.random() < self.tone_error_rate
                repl = _same_sound_char(ch, tone_err, rng)
                if repl:
                    chars[i] = repl
        return "".join(chars)

    def _same_sound_word(self, word: str, rng: random.Random) -> Optional[str]:
        """同音字逐字替换构造同音词（简化：只替换其中一个字）。"""
        idx = rng.randrange(len(word))
        repl = _same_sound_char(word[idx], False, rng)
        if repl:
            return word[:idx] + repl + word[idx + 1:]
        return None
