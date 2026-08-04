"""复读检测：归一化 + 相似度判定（agent 出口拦截与短期记忆去重共用）。

背景（2026-08-04 用户反馈）：bot 反复用同一话术，每条复读落进短期记忆后，
下一轮 context 里同一句话出现 N 次，模型把「自己老说这句」当成说话习惯
继续复读——上下文自我污染的正反馈循环。prompt 层劝不住，必须代码层
确定性处理：输入端渲染去重 + 出口端撞车拦截（agent.py echo guard）。

本模块放 junjun_memory 而非 junjun_agent/loop：short_term 要用它，
而 junjun_agent 依赖 junjun_memory，反向放会循环导入。
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional

_WS_PUNCT_RE = re.compile(r"[\s　，。！？!?、~…—\-·．.\"'“”‘’（）()【】\[\]<>《》:；;]+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)

_MIN_LEN = 4       # 归一化后短于 4 字不参与复读判定（「嗯嗯」「好耶」天然会重复）
_CONTAIN_LEN = 6   # 较短一方达到 6 字且被对方完整包含即判复读


def normalize_echo(text: str) -> str:
    """比较用归一化：去 emoji/标点/空白，小写。"""
    t = _EMOJI_RE.sub("", text or "")
    t = _WS_PUNCT_RE.sub("", t)
    return t.lower()


def is_echo(new_text: str, recent_texts: List[str], *,
            similarity: float = 0.85) -> Optional[str]:
    """新文本与近期发言撞车 -> 返回撞车的历史原文；否则 None。

    判定三档：归一化全等 / 6 字以上一方被另一方完整包含 / 相似度过阈。
    """
    new = normalize_echo(new_text)
    if len(new) < _MIN_LEN:
        return None
    for old_text in recent_texts:
        old = normalize_echo(old_text)
        if len(old) < _MIN_LEN:
            continue
        if new == old:
            return old_text
        if min(len(new), len(old)) >= _CONTAIN_LEN and (new in old or old in new):
            return old_text
        if SequenceMatcher(None, new, old).ratio() >= similarity:
            return old_text
    return None


def extract_catchphrases(texts: List[str], *, min_count: int = 3,
                         ngram_min: int = 4, ngram_max: int = 6) -> List[str]:
    """从 bot 近期发言里自动挖「口头禅」：在 >= min_count 条不同消息里
    出现过的 4~6 字片段（n-gram）。

    为什么需要它（2026-08-04 用户提问「示例集要一直维护吗」）：prompt 里的
    台词被复读是物理规律，靠人定期换示例是永续维护。改成代码自动检测——
    某个词组被用到第 3 次就进黑名单，agent 出口拦截（echo guard）会拦下
    任何再含它的回复，全程零人工。

    豁免：unique 字符 <= 2 的纯重复（哈哈哈哈/好好好好 这类天然会反复的）。
    返回按出现次数降序的口头禅列表（归一化形式，只保留最长代表）。
    """
    counts: dict = {}
    for text in texts:
        norm = normalize_echo(text)
        if len(norm) < ngram_min:
            continue
        seen = set()
        for n in range(ngram_min, ngram_max + 1):
            for i in range(len(norm) - n + 1):
                gram = norm[i:i + n]
                if len(set(gram)) <= 2:
                    continue
                seen.add(gram)
        for gram in seen:
            counts[gram] = counts.get(gram, 0) + 1
    hits = [(c, g) for g, c in counts.items() if c >= min_count]
    hits.sort(key=lambda x: (-x[0], -len(x[1])))
    result = []
    for _, g in hits:
        if not any(g in kept for kept in result):  # 被更长代表包含的短 gram 跳过
            result.append(g)
    return result[:10]
