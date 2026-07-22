"""回复后处理流水线：agent 原始文本 -> 多条待发消息。

顺序：去 <think> 残留 -> 分割多条 -> 错别字 -> （引用决策在 processor 层）。
纯函数，配置从 bot_config [response_post_process]/[response_splitter]/[chinese_typo] 读取。
"""

import random
import re
from dataclasses import dataclass
from typing import List, Optional

from junjun_core.config import get_global_config
from junjun_agent.postprocess.splitter import split_response, typing_delay
from junjun_agent.postprocess.typo import ChineseTypoGenerator

# 完整 <think>...</think> 块剥离
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
# 无闭合标签的尾部思考链（LLM 没吐 </think> 的情况）——从 <think> 到文本末尾全砍
_THINK_TAIL_RE = re.compile(r"<think>.*$", re.S)
_NICKNAME_PREFIX_RE = re.compile(r"^\s*(你|君君)\s*[:：]\s*")

_typo_gen: Optional[ChineseTypoGenerator] = None


@dataclass
class OutboundMessage:
    text: str
    delay: float = 0.0  # 发送前延迟（秒）


def _get_typo_gen() -> ChineseTypoGenerator:
    global _typo_gen
    if _typo_gen is None:
        cfg = get_global_config().raw.get("chinese_typo", {})
        _typo_gen = ChineseTypoGenerator(
            error_rate=float(cfg.get("error_rate", 0.01)),
            min_freq=int(cfg.get("min_freq", 9)),
            tone_error_rate=float(cfg.get("tone_error_rate", 0.1)),
            word_replace_rate=float(cfg.get("word_replace_rate", 0.006)),
        )
    return _typo_gen


def process_response(text: str, *, rand: Optional[random.Random] = None) -> List[OutboundMessage]:
    """agent 文本 -> 待发消息列表（含逐条打字延迟）。"""
    raw = get_global_config().raw
    pp = raw.get("response_post_process", {})
    sp = raw.get("response_splitter", {})
    typo_cfg = raw.get("chinese_typo", {})
    rng = rand or random

    text = _THINK_BLOCK_RE.sub("", text or "")
    text = _THINK_TAIL_RE.sub("", text)
    text = _NICKNAME_PREFIX_RE.sub("", text.strip())
    if not text:
        return []

    if not pp.get("enable_response_post_process", True):
        return [OutboundMessage(text=text)]

    pieces = split_response(
        text,
        enable=bool(sp.get("enable", True)),
        max_sentence_num=int(sp.get("max_sentence_num", 5)),
        max_chars_per_message=int(sp.get("max_chars_per_message", 120)),
        enable_kaomoji_protection=bool(sp.get("enable_kaomoji_protection", False)),
        enable_overflow_return_all=bool(sp.get("enable_overflow_return_all", True)),
        rand=rng,
    )

    if typo_cfg.get("enable", True):
        gen = _get_typo_gen()
        pieces = [gen.create_typo_sentence(p, rand=rng) for p in pieces]

    out: List[OutboundMessage] = []
    for i, p in enumerate(pieces):
        # 首条小延迟起步，后续按前一条字数模拟打字
        delay = typing_delay(pieces[i - 1], rand=rng) if i > 0 else 0.0
        out.append(OutboundMessage(text=p, delay=delay))
    return out
