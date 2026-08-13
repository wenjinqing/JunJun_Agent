"""全功能体检脚本：环境配置矩阵 + 插件装载审计 + 只读工具实调 + pytest 汇总。

用法：
    uv run python scripts/functional_check.py            # 全部检查（不含 pytest）
    uv run python scripts/functional_check.py --pytest   # 追加跑全套件
    uv run python scripts/functional_check.py --json     # 输出 JSON 到 stdout

安全约束：
- 只读实调：任何会发 QQ 消息/写库/花钱的工具一律 SKIP（标注原因）
- 生产库只读访问
- .env 只加载不打印（密钥绝不输出）
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []  # {category, item, status, latency_ms, note}


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def record(category: str, item: str, status: str, note: str = "",
           latency_ms: int = 0) -> None:
    RESULTS.append({"category": category, "item": item, "status": status,
                    "latency_ms": latency_ms, "note": note})


# ---------------------------------------------------------------- 1. 环境配置矩阵
# (功能, [(env或配置, 是否必需)], 说明)
ENV_MATRIX = [
    ("LLM 主模型(agent槽)", ["OPENAI_API_KEY|SILICONFLOW_API_KEY|DEEPSEEK_API_KEY"], "任一可用"),
    ("Langfuse 追踪", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"], "可缺（降级无追踪）"),
    ("pixiv", ["PIXIV_COOKIE"], "缺则全部 P 站功能不可用"),
    ("豆包 TTS", ["DOUBAO_TTS_API_KEY"], "四个后端任一即可"),
    ("硅基流动 TTS", ["SILICONFLOW_API_KEY"], ""),
    ("GSV2P TTS", ["TTS_GSV2P_TOKEN"], ""),
    ("SoVITS TTS", ["TTS_SOVITS_REF_AUDIO", "TTS_SOVITS_PROMPT_TEXT"], "本机服务"),
    ("AI 画图(ModelScope)", ["MODELSCOPE_API_KEY"], ""),
    ("谷歌搜索", ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"], "web_search 后备链"),
    ("天气", ["WEATHER_API_KEY|GAODE_API_KEY"], "视插件实现"),
    ("网管鉴权", ["GATEWAY_TOKEN"], "可缺（仅本机回环）"),
    ("管理员", ["ADMIN_QQ"], "根信任"),
]


def check_env() -> None:
    for item, keys, note in ENV_MATRIX:
        missing, present = [], []
        for k in keys:
            if "|" in k:
                alts = k.split("|")
                hit = next((a for a in alts if os.environ.get(a, "").strip()), None)
                (present if hit else missing).append(hit or k)
            else:
                (present if os.environ.get(k, "").strip() else missing).append(k)
        if not missing:
            record("环境配置", item, "PASS", note)
        elif present:
            record("环境配置", item, "DEGRADED",
                   f"缺 {','.join(missing)}（{note}）" if note else f"缺 {','.join(missing)}")
        else:
            record("环境配置", item, "FAIL", f"全缺：{','.join(k.replace('|','/') for k in keys)}")


# ---------------------------------------------------------------- 2. 插件装载审计
def check_plugins() -> list:
    from junjun_skills.registry import load_builtin
    from junjun_skills.plugin_loader import load_plugins
    load_builtin()
    n = load_plugins()
    tools = get_all_tools()
    record("插件装载", "插件加载数", "PASS", f"{n} 个插件，{len(tools)} 个 LLM 工具")

    # 目录有但日志没加载的插件（可能被合并/禁用/加载失败）
    plugins_dir = ROOT / "junjun_skills" / "plugins"
    dirs = sorted(d.name for d in plugins_dir.iterdir()
                  if d.is_dir() and not d.name.startswith("_"))
    return dirs


def get_all_tools() -> list:
    from junjun_skills.registry import get_tools
    return get_tools()


# ---------------------------------------------------------------- 3. 工具静态检查 + 只读实调
# 只读实调白名单：(工具名, 参数字典, 备注)。不在名单的一律 SKIP。
LIVE_PROBES = [
    ("get_time", {}, "系统时间"),
    ("use_skill", {"name": "video-watching"}, "技能手册读取(md skills)"),
    ("get_capabilities", {}, "能力自报"),
    ("query_jargon", {"term": "yyds"}, "黑话查询"),
    ("abbreviation_translate", {"term": "yyds"}, "缩写翻译(在线API)"),
    ("search_knowledge", {"question": "猫"}, "知识库检索"),
    ("today_in_history", {}, "历史上的今天(fun_texts)"),
    ("get_today_in_history", {}, "历史上的今天(news)"),
    ("answer_book", {"question": "今天顺利吗"}, "答案之书"),
    ("fun_quote", {}, "毒鸡汤"),
    ("draw_lot", {}, "抽签"),
    ("make_qrcode", {"text": "https://example.com"}, "生成二维码"),
    ("query_intimacy", {}, "好感度查询(需会话ctx)"),
    ("list_reminders", {}, "提醒列表"),
    ("list_subscriptions", {}, "订阅列表"),
    ("list_background_tasks", {}, "后台任务列表"),
    ("query_chat_history", {"keyword": "吃", "days": 7}, "聊天记录搜索(生产库只读)"),
    ("recall_memory", {"query": "猫"}, "记忆召回"),
    ("get_weather", {"city": "北京"}, "天气(在线API)"),
    ("web_search", {"query": "今日新闻"}, "网页搜索(在线API)"),
    ("pixiv_search_illusts", {"keyword": "原神"}, "P站插画搜索(在线)"),
    ("pixiv_search_novels", {"keyword": "原神"}, "P站小说搜索(在线)"),
    ("bilibili_summary", {"url": "https://www.bilibili.com/video/BV1GJ411x7h7"}, "B站视频摘要(在线)"),
    ("read_feed", {}, "读QQ空间(需NapCat在线)"),
    ("workspace_list", {}, "工作区列表(只读)"),
    ("fetch_page", {"url": "https://example.com"}, "网页深读(在线)"),
]

# 明确不实调的原因（写进报告）
SKIP_REASONS = {
    "do_not_reply": "沉默信号工具，无副作用但无验证价值",
    "save_memory": "写库", "pin_memory": "写库", "learn_jargon": "写库",
    "manage_user_profile": "写库", "set_reminder": "写库",
    "cancel_reminder_task": "写库", "manage_mood": "写全局心境",
    "import_knowledge": "写库", "send_message": "发QQ消息",
    "send_poke": "发QQ戳一戳", "send_emoji": "发QQ消息",
    "peek_group_chat": "需指定群号（手动验证）",
    "find_user_id": "需指定昵称（手动验证）",
    "ai_draw": "花钱(绘图API)", "deep_research": "长任务+花钱",
    "run_background_task": "派生后台任务", "cancel_background_task": "副作用",
    "bilibili_summary_cached": "不存在", "watch_video": "派生后台任务+下载",
    "chat_screenshot": "渲染截图（pytest已覆盖）",
    "query_cross_scene_chat": "跨场景查询（pytest已覆盖）",
    "douyin_summary": "需真实分享链接（手动验证）",
    "decode_qrcode": "需图片输入（pytest已覆盖）",
    "ja_tts": "花钱(TTS API)", "unified_tts": "花钱(TTS API)",
    "send_feed": "发QQ空间说说", "delete_feed": "删说说",
    "play_music": "发QQ消息", "pixiv_send_illust": "发QQ消息",
    "pixiv_download_novel": "发QQ消息",
    "subscribe_updates": "写库", "unsubscribe": "写库",
    "run_code": "需沙箱服务+管理员门禁（pytest 已覆盖门禁/预检）",
    "workspace_read": "需指定文件（pytest 已覆盖）",
    "workspace_write": "写工作区文件（pytest 已覆盖）",
}


async def probe_tools() -> None:
    from junjun_skills.builtin.memory_skills import current_chat_id
    tools = {t.name: t for t in get_all_tools()}

    # 3a. 静态：每个工具都要有 docstring 和合法 schema
    for name, t in sorted(tools.items()):
        doc = (t.description or "").strip()
        if not doc:
            record("工具静态", name, "FAIL", "缺 docstring——模型选工具全靠它")
        elif len(doc) < 15:
            record("工具静态", name, "DEGRADED", f"docstring 过短({len(doc)}字)")
        else:
            record("工具静态", name, "PASS", f"doc {len(doc)}字")

    # 3b. 只读实调
    token = current_chat_id.set("qq:10000001:private")
    try:
        for name, args, note in LIVE_PROBES:
            t = tools.get(name)
            if t is None:
                record("工具实调", name, "FAIL", f"工具不存在（{note}）")
                continue
            t0 = time.time()
            try:
                out = await asyncio.wait_for(t.ainvoke(args), timeout=45)
                text = str(out)
                latency = int((time.time() - t0) * 1000)
                low = text.lower()
                if "未配置" in text or "缺少" in text or "没有配置" in text:
                    record("工具实调", name, "DEGRADED", text[:80], latency)
                elif "error" in low[:30] or "失败" in text[:30]:
                    record("工具实调", name, "FAIL", text[:80], latency)
                else:
                    record("工具实调", name, "PASS",
                           f"{note}；返回 {len(text)} 字符", latency)
            except asyncio.TimeoutError:
                record("工具实调", name, "FAIL", f"超时(45s)（{note}）")
            except Exception as e:
                record("工具实调", name, "FAIL",
                       f"{type(e).__name__}: {str(e)[:80]}（{note}）")
    finally:
        current_chat_id.reset(token)

    # 3c. 不实调的记录 SKIP
    probed = {n for n, _, _ in LIVE_PROBES}
    for name in sorted(tools):
        if name in probed:
            continue
        record("工具实调", name, "SKIP",
               SKIP_REASONS.get(name, "有副作用，不实调"))


# ---------------------------------------------------------------- 4. LLM 链路活性
async def probe_llm() -> None:
    for slot in ("utils", "agent"):
        t0 = time.time()
        try:
            from junjun_llm import get_chat_model
            model = get_chat_model(slot)
            resp = await asyncio.wait_for(
                model.ainvoke([{"role": "user", "content": "只回复两个字：正常"}]),
                timeout=30)
            text = str(getattr(resp, "content", resp))[:30]
            record("LLM链路", f"{slot}槽", "PASS",
                   f"响应正常（{text[:12]}）", int((time.time() - t0) * 1000))
        except Exception as e:
            record("LLM链路", f"{slot}槽", "FAIL", f"{type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------- 5. pytest 汇总
def run_pytest() -> None:
    import subprocess
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    tail = (p.stdout or "").strip().splitlines()[-1] if p.stdout else p.stderr[-200:]
    ok = p.returncode == 0
    record("pytest", "全套件", "PASS" if ok else "FAIL", tail,
           int(time.time() - t0))


# ---------------------------------------------------------------- 输出
def to_markdown() -> str:
    lines = ["| 类别 | 项目 | 状态 | 耗时ms | 备注 |", "|---|---|---|---|---|"]
    for r in RESULTS:
        note = str(r["note"]).replace("|", "/").replace("\n", " ")[:100]
        lines.append(f"| {r['category']} | {r['item']} | {r['status']} "
                     f"| {r['latency_ms'] or '-'} | {note} |")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    _load_env()
    check_env()
    check_plugins()
    await probe_tools()
    await probe_llm()
    if args.pytest:
        run_pytest()

    if args.json:
        print(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    else:
        summary = {}
        for r in RESULTS:
            summary[r["status"]] = summary.get(r["status"], 0) + 1
        print(to_markdown())
        print(f"\n汇总: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
