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
    # 调研链：检索 × 产出。
    # A 组只放强调研词——「查一下/搜一下」曾入列，「查一下体检报告出来了没」
    # 这类单步查询全被拆步骤（2026-08-06 审查实锤）；B 组同理不收「摘要」
    # （作宾语太常见），产出诉求由 写份/写成/整理成 等动词表达。
    # 2026-08-13 加宽（B 组补重产出名词）：「调研/搜集…清单/简报/笔记/要点/文档」
    # ——A 组本身是强调研词，配上显式产出物的委托对话通道装不下。
    (("调研", "研究一下", "查资料", "深度搜索", "搜集"),
     ("报告", "写份", "写成", "整理成", "汇总成", "梳理成", "综述",
      "清单", "简报", "笔记", "要点", "文档")),
    # 阅读链：深读 × 产出（2026-08-13 A 补「翻翻」、B 补「总结/清单」：
    # 「翻翻咱们这周都聊过什么，给我个总结」——翻历史+蒸馏两步）
    (("看完", "读一下", "读这篇", "仔细看", "精读", "读完", "翻翻"),
     ("整理", "笔记", "汇总", "讲给", "梳理", "报告", "总结一下", "提炼",
      "总结", "清单")),
    # 检索 × 成品/产出（2026-08-13 B 组扩重产出名词 + 「整理」：单问查询
    # 不带这些宾语，「搜一下最新政策整理要点给我」是检索+加工两步）
    (("查一下", "搜一下", "找一下", "查查", "搜搜", "搜最近", "查最近"),
     ("画一", "画张", "画个", "生成图", "做成图",
      "清单", "简报", "综述", "要点", "整理")),
    # 获取/成品 × 发布（发布必须走任务通道才有人审门——对话通道直发没门。
    # 2026-08-13 A 组补 搜/查/看个：「搜新闻写综述发到空间」「看个视频把感想
    # 发到空间」此前漏路由，说说绕过人审直发）
    (("画一", "画张", "画个", "生成", "搜", "查", "看个", "看一"),
     ("发空间", "发说说", "发到空间", "发条说说", "发到说说")),
]

# 显式重产出委托（2026-08-13「写一份研究笔记」事故信号——疯狂搜索第一环）：
# 产出动词+量词+重产出名词 = 要调研/合成多步的委托；名词清单保守，
# 「写个段子」「出个主意」这类轻产出不收（对话通道单轮合成足够）。
_PRODUCE_RE = re.compile(r"(写|做|整|整理|出|生成)\s*一?\s*(份|篇|个|张|版)")
_HEAVY_NOUNS = ("报告", "笔记", "综述", "简报", "攻略", "文档", "清单", "方案")

# 查询 × 提醒承接链：「查一下明天天气，如果下雨就提醒我带伞」——「未来触发」
# 对话通道单轮语义装不下（提醒工具只管「明天提醒我开会」这种单步）。
# 不收「盯」：盯梢/订阅是对话通道 subscribe_updates 的专属地盘（「帮我盯一下
# UP主，更新了告诉我」派成一次性任务是误路由，2026-08-13 golden_cases 实锤）。
_CHAIN_TAILS = ("提醒我", "告诉我", "到时候叫", "到时候通知")
_QUERY_WORDS = ("查", "搜", "看看", "看一下")

# 文件加工链：文件来源信号 + 加工动作（「把 notes.md 转成表格存回去」）。
# 「刚发」必须配文件类名词——「我刚发工资了存起来」是日常不是派单。
_FILE_SOURCES = ("工作区", "发的文件", "发来的文件", "这个文件", "这个表格",
                 ".md", ".csv", ".pdf", ".xlsx", ".docx", ".txt", ".pptx")
_FILE_NOUNS = ("文件", "表格", "文档", "压缩包", "照片")
_FILE_ACTIONS = ("转成", "转为", "存", "统计", "算", "画", "整理", "汇总",
                 "分析", "解读", "读一下", "看看", "看一下", "总结")

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
    # 显式重产出委托
    if _PRODUCE_RE.search(t) and any(n in t for n in _HEAVY_NOUNS):
        logger.debug(f"路由->任务通道（重产出委托）: {t[:30]}")
        return True
    # 查询 × 提醒承接链
    if (any(w in t for w in _QUERY_WORDS)
            and any(w in t for w in _CHAIN_TAILS)):
        logger.debug(f"路由->任务通道（查询+提醒链）: {t[:30]}")
        return True
    # 文件加工链
    has_file = any(w in t for w in _FILE_SOURCES) or (
        "刚发" in t and any(n in t for n in _FILE_NOUNS))
    if has_file and any(w in t for w in _FILE_ACTIONS):
        logger.debug(f"路由->任务通道（文件加工链）: {t[:30]}")
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


# ---- 主 Agent 双腿路由（2026-08-11 token 优化 P2）----
# agent 槽独占约 99% token 消耗（生产实测：单次平均输入 14.5K，固定前缀
# system+工具占一半以上）。纯闲聊轮不需要 ***REMOVED*** 的工具调用稳定性，
# 走 ***REMOVED*** 轻腿（agent_light 槽）省输入溢价。
# 判 light 是「加宽命中面」，纪律照旧：条件【全部】命中才放行，拿不准一律
# full（现状腿）——误判方向只亏钱（该轻的走了贵的），绝不亏人格与工具可靠性。
# 配套误判回归测试在 tests/test_agent_tier.py。

# 轻腿长上限：纯闲聊轮实测绝大多数 < 50 字；长消息往往带信息密度/多诉求
_TIER_LIGHT_MAX_CHARS = 50

# 事实/时效信号：persona 规则强约束这些必须先调 web_search——工具调用
# 可靠性正是轻腿的短板，直接走强腿
_TIER_FACT_WORDS = ("什么时候", "最新", "天气", "新闻", "几点", "多少钱")

# 意图组词表之外的工具诉求补充（误判回归实锤：「定个闹钟」不含「提醒我」，
# 意图自检同款盲区——路由层不能再漏）
_TIER_TOOL_WORDS = ("闹钟", "定时", "翻译一下", "帮我翻译")


def agent_tier(text: str, *, has_media: bool = False) -> str:
    """本轮主 Agent 用哪条腿："light"（闲聊轻腿）/ "full"（默认强链）。

    灰度开关：[agent] complexity_routing（默认 false = 永远 full，回到现状）。
    """
    try:
        from junjun_core.config import get_global_config
        if not bool(get_global_config().raw.get("agent", {})
                    .get("complexity_routing", False)):
            return "full"
    except Exception:
        return "full"
    t = (text or "").strip()
    if not t or len(t) > _TIER_LIGHT_MAX_CHARS:
        return "full"
    if has_media:
        return "full"           # 图/语音/视频轮：感知块+工具链，走强腿
    if "http" in t or "www." in t:
        return "full"           # 链接：拦截器没吃完的往往要工具理解
    try:
        from junjun_core.security import is_admin_privileged
        if is_admin_privileged():
            return "full"       # 管理员的拜托走强腿（敏感操作可靠性）
    except Exception:
        pass
    try:
        from junjun_skills.registry import intent_groups
        for kws, _group, _primary in intent_groups():
            if any(kw in t for kw in kws):
                return "full"   # 强意图（订阅/提醒/搜/画/语音/调研）：工具轮
    except Exception:
        return "full"           # 元数据拿不到时保守走强腿
    if any(w in t for w in _TIER_FACT_WORDS):
        return "full"
    if any(w in t for w in _TIER_TOOL_WORDS):
        return "full"
    try:
        from junjun_agent.loop.plan_tracker import detect_complexity
        if detect_complexity(t):
            return "full"       # 疑似复合任务
    except Exception:
        pass
    if route_to_task(t):
        return "full"           # 任务通道回退到对话的轮次，走强腿
    return "light"
