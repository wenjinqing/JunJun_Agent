"""路由层（架构重写方案 §4.2）：复杂任务 -> 任务通道（TaskKernel），其余 -> 对话通道。

0-token 严格规则，宁漏勿错：
- 漏判 = 走现状对话通道（单轮 agent + 工具循环），无回归；
- 误判 = 简单请求被拆步骤（成本+绕路+奇怪的接单话术）。
所以只认强信号：异质动作对 / 时程承诺 / 显式多步。
拿不准的一律 chat——LLM 分类兜底是后续增强，v1 先纯规则。
"""

import re

from junjun_core.observability import get_logger

logger = get_logger("agent.router")

# 异质动作对：两个不同组的动作词同时出现才算复杂。
# 「查一下天气」是单步（只有查），「查资料写成报告」才是任务（查+写）。
_ACTION_PAIRS = [
    # 调研链：检索 × 产出
    (("调研", "研究一下", "查资料", "深度搜索", "搜集"),
     ("报告", "写份", "写成", "整理成", "汇总成", "梳理成", "综述")),
    # 阅读链：深读 × 产出
    (("看完", "读一下", "读这篇", "仔细看", "精读", "读完"),
     ("整理", "笔记", "汇总", "讲给", "梳理", "报告", "总结一下", "提炼")),
    # 检索 × 成品
    (("查一下", "搜一下", "找一下", "查查", "搜搜"),
     ("画一", "画张", "画个", "生成图", "做成图")),
    # 成品 × 发布
    (("画一", "画张", "画个", "生成"),
     ("发空间", "发说说", "发到空间", "发条说说")),
]

# 时程承诺：单次请求里要求周期性/持续性执行——对话通道的单轮语义根本装不下
_SCHEDULE_WORDS = ("每天早上", "每天晚上", "每天给我", "每天帮我", "每周",
                   "持续", "定期", "以后每天", "长期")
# 「天天」单独处理：「明天天气」误伤过（子串撞车）——前面是 明/今/昨/天 时不算
_TIANTIAN_EXCLUDE_PREFIX = ("明", "今", "昨", "天")

# 显式多步信号：连接词两侧都有动作词（「先查一下然后画出来」）
_STEP_JOINS = ("然后", "接着", "之后", "再顺", "最后再", "然后再")
_ACTION_WORDS = ("查", "搜", "找", "画", "写", "整理", "总结", "翻译", "下载",
                 "调研", "分析", "对比", "生成", "看完", "读完")

# 否定句排除：「别帮我调研」「不用写报告」是制止不是派单
_NEGATIONS = ("别", "不要", "不用", "不许", "不准", "算了", "先不")

_Q_RE = re.compile(r"[吗呢啊？?]$")


def route_to_task(text: str, *, chat_id: str = "") -> bool:
    """强信号命中 -> True（任务通道）。拿不准一律 False（对话通道）。"""
    t = (text or "").strip()
    if not t or len(t) < 6:
        return False
    # 纯疑问句不派单（「调研报告怎么写」是请教不是委托）
    if _Q_RE.search(t) and not any(w in t for w in ("帮我", "给我", "麻烦", "请你")):
        return False
    # 否定句：「别调研了」「不用写报告」是制止
    for neg in _NEGATIONS:
        if neg in t:
            return False
    # 时程承诺
    if any(w in t for w in _SCHEDULE_WORDS):
        logger.debug(f"路由->任务通道（时程承诺）: {t[:30]}")
        return True
    idx = t.find("天天")
    while idx != -1:
        if idx == 0 or t[idx - 1] not in _TIANTIAN_EXCLUDE_PREFIX:
            logger.debug(f"路由->任务通道（时程承诺「天天」）: {t[:30]}")
            return True
        idx = t.find("天天", idx + 1)
    # 异质动作对
    for group_a, group_b in _ACTION_PAIRS:
        if any(w in t for w in group_a) and any(w in t for w in group_b):
            logger.debug(f"路由->任务通道（动作对 {group_a[0]}×{group_b[0]}）: {t[:30]}")
            return True
    # 显式多步：连接词切开，两侧各含动作词
    for join in _STEP_JOINS:
        if join in t:
            left, right = t.split(join, 1)
            if (any(w in left for w in _ACTION_WORDS)
                    and any(w in right for w in _ACTION_WORDS)):
                logger.debug(f"路由->任务通道（显式多步「{join}」）: {t[:30]}")
                return True
    return False
