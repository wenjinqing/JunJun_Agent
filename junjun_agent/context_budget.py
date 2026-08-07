"""上下文预算：统一 token 配额与按优先级驱逐（架构重写 Phase 2）。

把进入 agent 的 prompt 拆成若干 BudgetBlock（system/情绪/记忆/关系/背景/最新消息），
每块申报优先级与估算 token，超预算时从低优先级、非必需块开始驱逐，保证高优先级
块（system、最新消息、安全锚点）始终在位。

设计要点：
- 估算不依赖 tiktoken：用字符/词混合启发式，对中英文混合场景足够准。
- 驱逐顺序：priority 数字小优先；同 priority 先驱逐体积大的（收益高）。
- 必需块（required=True）永不驱逐；如果必需块自身超预算，原样保留并告警。
- 结果写 Langfuse metadata：各块原始/最终大小、被驱逐块名。
"""

from dataclasses import dataclass, field
from typing import List, Optional

from junjun_core.observability import get_logger

logger = get_logger("context_budget")


@dataclass
class BudgetBlock:
    """上下文预算单元。"""
    name: str                       # 块标识，如 "system"
    content: str                    # 块文本
    priority: int                   # 优先级，数字越小越重要（1=最高）
    estimated_tokens: int = 0       # 预估算 token；0 时由 budget 自动估算
    required: bool = False          # 是否必需（永不驱逐）


@dataclass
class BudgetFit:
    """fit() 结果。"""
    blocks: List[BudgetBlock]       # 保留下来的块（按原传入顺序）
    evicted: List[BudgetBlock]      # 被驱逐的块
    total_tokens: int               # 保留块总 token
    max_tokens: int                 # 预算上限
    metrics: dict = field(default_factory=dict)  # 原始尺寸、驱逐原因等


class ContextBudget:
    """上下文预算管理器。"""

    def __init__(self, max_total_tokens: int, reserve_tokens: int = 0):
        self.max_total_tokens = max(1, max_total_tokens)
        self.reserve_tokens = max(0, reserve_tokens)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """快速估算 token 数（无 tiktoken 依赖）。

        cl100k_base 实测约 1.5 个中文字符/token、4 个英文字符/token。
        曾按「中英文估算取 min」实现——中文无空格，text.split() 把全文算成
        1 个词，609 字中文块估出 33 token（低估 10 倍），驱逐逻辑永不触发、
        Langfuse 指标全线失真（2026-08-06 审查实锤）。宁高估勿低估：
        高估只是提前驱逐，低估是预算形同虚设。
        """
        if not text:
            return 0
        cjk_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
        non_cjk_len = len(text) - cjk_chars
        return max(1, int(cjk_chars / 1.5 + non_cjk_len / 4.0))

    def fit(self, blocks: List[BudgetBlock]) -> BudgetFit:
        """按预算保留块，返回 fit 结果。

        步骤：
        1. 自动填充 estimated_tokens=0 的块。
        2. 若所有块（含必需）不超过预算，全部保留。
        3. 否则按 priority 升序、同 priority 体积降序排序，逐个尝试保留；
           必需块强制保留，非必需块空间不够则驱逐。
        """
        # 复制一份，避免修改传入对象
        work = [BudgetBlock(
            name=b.name, content=b.content, priority=b.priority,
            estimated_tokens=b.estimated_tokens or self.estimate_tokens(b.content),
            required=b.required,
        ) for b in blocks]

        effective_budget = max(1, self.max_total_tokens - self.reserve_tokens)

        required = [b for b in work if b.required]
        required_tokens = sum(b.estimated_tokens for b in required)
        if required_tokens > self.max_total_tokens:
            logger.warning(f"上下文必需块已超总预算: {required_tokens} > {self.max_total_tokens}，"
                           f"保留并告警（请检查 system prompt 体积）")

        # 非必需块按优先级、体积排序：优先保小的重要块，驱逐大块的低优先级块
        optional = [b for b in work if not b.required]
        optional.sort(key=lambda b: (b.priority, -b.estimated_tokens))

        kept = list(required)
        evicted: List[BudgetBlock] = []
        total = required_tokens
        # 先尝试所有必需块，必需块之间也受预算限制但强制保留
        # 非必需块在剩余空间里按优先级填入
        for b in optional:
            if total + b.estimated_tokens <= effective_budget:
                kept.append(b)
                total += b.estimated_tokens
            else:
                evicted.append(b)

        # 保持原始顺序：按传入 blocks 的顺序输出 kept
        kept_names = {b.name for b in kept}
        kept_in_order = [b for b in work if b.name in kept_names]

        metrics = {
            "max_total_tokens": self.max_total_tokens,
            "reserve_tokens": self.reserve_tokens,
            "effective_budget": effective_budget,
            "original_total_tokens": sum(b.estimated_tokens for b in work),
            "kept_total_tokens": total,
            "evicted_total_tokens": sum(b.estimated_tokens for b in evicted),
            "evicted_names": [b.name for b in evicted],
            "block_sizes": {b.name: b.estimated_tokens for b in work},
        }

        return BudgetFit(
            blocks=kept_in_order,
            evicted=evicted,
            total_tokens=total,
            max_tokens=self.max_total_tokens,
            metrics=metrics,
        )

    def build_messages(self, blocks: List[BudgetBlock]) -> tuple[List[BudgetBlock], dict]:
        """便捷方法：直接返回保留块 + metrics。

        供 agent.py 使用；返回的 blocks 已按原顺序保留，metrics 可写入 Langfuse。
        """
        fit = self.fit(blocks)
        if fit.evicted:
            logger.info(f"上下文驱逐: {fit.metrics['evicted_names']}，"
                        f"保留 {fit.total_tokens}/{self.max_total_tokens} tokens")
        return fit.blocks, fit.metrics
