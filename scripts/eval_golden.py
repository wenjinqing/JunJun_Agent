"""Golden dataset 评测运行器（严厉审查 P0-1 处方）。

用法：
    uv run python scripts/eval_golden.py              # 全量跑（真实 LLM，花 API 额度）
    uv run python scripts/eval_golden.py --only draw  # 只跑 id 含 draw 的 case
    uv run python scripts/eval_golden.py --report     # 只打印最近一次报告

设计：
- 用真实 agent 槽位模型 + 真实 system prompt + 真实工具 schema（名字/描述/参数
  原样保留），但工具执行体换成记录桩——评测「决策质量」（该调什么工具、该不该
  沉默、话术诚实度），不评测工具本身，也不产生副作用（不碰生产库/不发消息）。
- 判定全部确定性：must_call（「a|b」表示任一）、must_not_call、silence、
  reply_required、must_contain、must_not_contain。确定性优先于 LLM-judge
  （OpenAI evals 指南：能确定判定就不要上 judge）。
- 结果写 data/eval_report_<ts>.json，对比历史报告看回归。

注意：这是决策层评测（全量工具可用），工具掩码层的评测是下一阶段。
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CASES_FILE = ROOT / "tests" / "eval" / "golden_cases.jsonl"
REPORT_DIR = ROOT / "data"

# 各工具的桩返回值：要足够像真话，模型才能正常续写最终回复
_STUB_RETURNS = {
    "ai_draw": "图片任务已接受，正在生成，完成后会自动发送给对方。",
    # 桩内容必须像真新闻：占位/示例字样的假数据会被模型识破并拒绝二次利用
    # （2026-08-06 search-post-feed 实锤：模型说「占位信息我可不能随便发你空间」——
    # 行为是对的，是桩 sabotage 了 case）
    "web_search": "（桩）搜索结果：1. 微博热搜第一：某品牌发布新一代折叠屏手机，起售价 8999 元，网友热议铰链设计。2. 八部门印发人工智能产业高质量发展行动方案。3. 国家天文台发布最新系外行星观测成果。",
    "mcp_search": "（桩）搜索结果：1. 微博热搜第一：某品牌发布新一代折叠屏手机，起售价 8999 元，网友热议铰链设计。2. 八部门印发人工智能产业高质量发展行动方案。3. 国家天文台发布最新系外行星观测成果。",
    "get_time": "2026-08-04 15:30 星期二",
    "get_weather": "北京 明天 晴，25~33℃，微风。",
    "set_reminder": "提醒已设置成功，到点会叫你。",
    # 桩格式必须与生产 list_reminders 输出一致（[hex编号]（每天）MM月DD日 HH:MM），
    # 桩教模型一种生产里不存在的格式 = 评测在练错误动作（2026-08-09 审查实锤）
    "list_reminders": "（桩）待办提醒：\n- [a1b2c3] （每天）08月09日 08:00 推送科技新闻\n- [d4e5f6] 08月10日 20:00 抢票",
    "cancel_reminder_task": "提醒已取消。",
    "list_background_tasks": "（桩）后台任务：1 个进行中 [id=j7]（视频观看，已 12 分钟）；0 个已完成。",
    "cancel_background_task": "后台任务已取消。",
    "subscribe_updates": "订阅已创建成功，有更新会推送。",
    "unsubscribe": "已取消该订阅。",
    "unified_tts": "语音已合成并发送。",
    "ja_tts": "日语语音已合成并发送。",
    "play_music": "歌曲已开始播放。",
    "send_feed": "说说已发布。",
    "read_feed": "（桩）好友空间动态：1. 甲：今天去了海边。2. 乙：新游戏真好玩。",
    "bilibili_summary": "（桩）视频内容摘要：该视频讲解了主题A的三个要点……",
    "douyin_summary": "（桩）视频内容摘要：这是一条关于主题B的短视频……",
    "watch_video": "已派到后台仔细观看，看完会主动汇报观后感。",
    "deep_research": "调研任务已派到后台，完成后会主动把报告发给你。",
    "run_background_task": "任务已派到后台执行，完成后会主动汇报。",
    "save_memory": "已记住。",
    "pin_memory": "已钉住该记忆。",
    "recall_memory": "（桩）召回结果：未找到相关记忆。",
    "peek_group_chat": "（桩）该群最近聊天：大家在讨论新游戏发售。",
    "query_cross_scene_chat": "（桩）查询结果：该群最近在聊新游戏。",
    "list_subscriptions": "（桩）当前订阅：1. [id=1] UP主影视飓风（bilibili）；2. [id=2] P站画师mignon。",
    "do_not_reply": "已沉默。",
}
_DEFAULT_STUB_RETURN = "（桩）操作成功。"

# introduce_self 桩：必须与生产输出同构（速写+能力概览+技术栈段）。技术栈段
# 直接引生产文案单一数据源，防桩与生产漂移——占位「操作成功」桩会让模型
# 如实说「系统没把底细递给我」，回复内容断言永远挂（2026-08-15
# identity-who-are-you 实锤：桩 sabotage 了 case，同 list_reminders 格式教训）
try:
    from junjun_skills.builtin.capability_skills import _INTRO_TECH
except Exception:
    _INTRO_TECH = ("技术栈与架构：Python 写的；决策内核跑在 LangChain + LangGraph 上；"
                   "具体模型型号不公开。")
_STUB_RETURNS["introduce_self"] = (
    "我是君君——群里的猫娘学姐：从容温柔，会笑着调侃人。\n"
    "本体是个跑在服务器上的 AI 程序（被问起就大方承认），目前挂着 60 件工具。\n"
    "平时能干的：联网搜索、深度调研与后台长任务、B 站视频、AI 画图、语音合成、"
    "点歌放歌、工作区（收发文件/跑代码）、QQ 空间发说说、新闻速览。\n"
    "内置基本功：设提醒、记事回忆、查天气、翻聊天记录、发表情。\n"
    + _INTRO_TECH + "\n"
    "想看逐条的完整能力清单就调 get_capabilities。"
)


def _make_stub_tools():
    """真实工具 schema + 记录桩执行体。返回 (stub_tools, called_list)。"""
    from langchain_core.tools import StructuredTool
    from junjun_skills.registry import get_tools as real_get_tools

    called = []  # [(name, args)]

    stubs = []
    for t in real_get_tools():
        name = t.name

        async def _stub(_name=name, **kwargs):
            called.append((_name, kwargs))
            return _STUB_RETURNS.get(_name, _DEFAULT_STUB_RETURN)

        stubs.append(StructuredTool(
            name=name,
            description=t.description or "",
            args_schema=getattr(t, "args_schema", None),
            coroutine=_stub,
        ))
    return stubs, called


async def _run_case(agent_mod, stubs, called, case: dict) -> dict:
    from junjun_agent.agent import JunJunAgent

    called.clear()
    scene = case.get("scene", "private")
    session = SimpleNamespace(
        chat_id=f"eval:{case['id']}:{scene}",
        is_group=(scene == "group"),
        memory=None,
    )
    agent = JunJunAgent(session)

    nickname = "小明"
    latest_line = f"「{nickname}」: {case['input']}"
    if case.get("addressed") and scene == "group":
        latest_line = f"「{nickname}」 [@你]: {case['input']}"
    bg = (case.get("background") or "").strip()
    context_text = (bg + "\n" if bg else "") + latest_line

    try:
        reply = await asyncio.wait_for(
            agent.process(
                context_text,
                latest_text=case["input"],
                addressed=bool(case.get("addressed")),
                memory_block=case.get("memory_block", ""),
                trace_id=f"eval-{case['id']}",
            ),
            timeout=240,
        )
    except asyncio.TimeoutError:
        return {"id": case["id"], "pass": False, "reason": "TIMEOUT(240s)"}
    except Exception as e:
        return {"id": case["id"], "pass": False,
                "reason": f"ERROR {type(e).__name__}: {e}"}
    finally:
        try:
            await agent.aclose()
        except Exception:
            pass

    tools_called = [n for n, _ in called]
    exp = case.get("expect", {})
    fails = []

    for spec in exp.get("must_call", []):
        alts = spec.split("|")
        if not any(a in tools_called for a in alts):
            fails.append(f"未调用 {'/'.join(alts)}（实际: {tools_called or '无'}）")
    for name in exp.get("must_not_call", []):
        if name in tools_called:
            fails.append(f"不应调用 {name}")
    # 参数断言（2026-08-09 审查）：工具名对了参数不对一样是错——
    # send_feed 不带 with_image=空头说说；set_reminder 的 time_spec 没「每天」
    # =周期承诺落成了一锤子买卖。字符串按子串匹配，布尔/数字按相等。
    for spec in exp.get("must_call_args", []):
        t_name, want = spec["tool"], spec.get("args", {})

        def _match(actual, w):
            if actual is None:
                return False
            if isinstance(w, str):
                return w in str(actual)
            return actual == w

        hit = any(n == t_name and all(_match(a.get(k), w) for k, w in want.items())
                  for n, a in called)
        if not hit:
            got = [a for n, a in called if n == t_name]
            fails.append(f"{t_name} 参数不符（期望含 {want}，实际: {got or '未调用'}）")
    if exp.get("silence"):
        if reply is not None and "do_not_reply" not in tools_called:
            fails.append(f"应沉默却回复: {str(reply)[:40]}")
    if exp.get("reply_required") and not reply:
        fails.append(f"应回复却沉默（工具: {tools_called or '无'}）")
    text = reply or ""
    for s in exp.get("must_contain", []):
        if s not in text:
            fails.append(f"回复缺少「{s}」: {text[:60]}")
    for s in exp.get("must_not_contain", []):
        if s in text:
            fails.append(f"回复不应含「{s}」: {text[:60]}")

    return {
        "id": case["id"], "pass": not fails,
        "reason": "；".join(fails),
        "tools": tools_called,
        "reply": (text[:200] if text else None),
    }


async def _main(args) -> int:
    cases = [json.loads(l) for l in
             CASES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if not cases:
        print("没有匹配的 case")
        return 1

    from junjun_skills.registry import load_builtin
    from junjun_skills.plugin_loader import load_plugins
    load_builtin()
    load_plugins()

    stubs, called = _make_stub_tools()

    # 接管 agent 的工具获取与副作用点
    import junjun_agent.agent as agent_mod
    agent_mod.get_tools = lambda *a, **kw: list(stubs)
    agent_mod._record_usage = lambda *a, **kw: None
    import junjun_skills.registry as reg
    reg.warm_tool_embeddings = lambda *a, **kw: asyncio.sleep(0)

    results = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        r = await _run_case(agent_mod, stubs, called, case)
        results.append(r)
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{i}/{len(cases)}] {mark} {r['id']}"
              + ("" if r["pass"] else f"  -- {r['reason']}"))
        # 实时落盘，中途挂了也有部分结果
        _write_report(results, t0)

    passed = sum(1 for r in results if r["pass"])
    print(f"\n==== {passed}/{len(results)} 通过，耗时 {time.time()-t0:.0f}s ====")
    print(f"报告: {_write_report(results, t0)}")
    return 0 if passed == len(results) else 1


def _write_report(results: list, t0: float) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t0))
    path = REPORT_DIR / f"eval_report_{ts}.json"
    passed = sum(1 for r in results if r["pass"])
    path.write_text(json.dumps({
        "ts": ts, "passed": passed, "total": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # latest 软链式副本，方便 diff
    (REPORT_DIR / "eval_report_latest.json").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _show_latest() -> int:
    p = REPORT_DIR / "eval_report_latest.json"
    if not p.exists():
        print("还没有评测报告")
        return 1
    rep = json.loads(p.read_text(encoding="utf-8"))
    print(f"报告 {rep['ts']}: {rep['passed']}/{rep['total']}")
    for r in rep["results"]:
        if not r["pass"]:
            print(f"  FAIL {r['id']} -- {r['reason']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑 id 包含该子串的 case")
    ap.add_argument("--report", action="store_true", help="只看最近一次报告")
    args = ap.parse_args()
    if args.report:
        sys.exit(_show_latest())
    sys.exit(asyncio.run(_main(args)))
