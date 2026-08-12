"""golden_tasks 任务向评测运行器（Phase 0：一切 Phase 的度量依据）。

用法：
    uv run python scripts/eval_tasks.py               # 全量跑（真实 LLM，花 API 额度）
    uv run python scripts/eval_tasks.py --only feed   # 只跑 id 含 feed 的 case
    uv run python scripts/eval_tasks.py --report      # 只看最近一次报告
    uv run python scripts/eval_tasks.py --baseline    # 跑完另存基线存档

设计（照 eval_golden.py 骨架，判定从「调了什么工具」升级为三层）：
- 真实 planner/执行器（langgraph 引擎）+ 真实 LLM；工具执行体换记录桩——
  评测「任务到底办没办成」，不产生副作用（不发消息/不碰生产库）。
- 三层判定：
  1. 完成判定：plan.state == done（代码可查，0 token）
  2. 产物判定：工具调用记录/步骤状态/结果包含断言（确定性优先）
  3. 质量抽检：llm_judge 1-5 分（仅 judge:true 的产物类 case，控制成本）
- 指标三件套：成功率 / 平均步数 / token 成本（响应 usage 逐调用累计）。
- 人审模拟：approval=approve/reject 由脚本扮演管理员 resume；
  approval=timeout 不响应，靠 approval_timeout_seconds 看门狗默认跳过。
- 闲聊负例：try_submit 必须返回 None（不接单），防内核过度接单误伤对话。

case 格式见 tests/eval/golden_tasks.jsonl；enabled:false 的占位 case 自动跳过。
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

# 评测进程强制关 Langfuse：token 成本由 _Usage 从响应 usage 自记，不依赖
# trace；本机 localhost:3000 代理半死时导出还会拖慢调用。只影响评测进程，
# 不动 .env。（退出挂死的真凶是 aiosqlite，见下方 MemorySaver 补丁注释。）
os.environ["LANGFUSE_ENABLED"] = "false"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CASES_FILE = ROOT / "tests" / "eval" / "golden_tasks.jsonl"
REPORT_DIR = ROOT / "data"
BASELINE_FILE = REPORT_DIR / "eval_tasks_baseline_20260812.json"

_EVAL_CFG = {
    "enable": True,
    "engine": "langgraph",
    "max_steps": 6,
    "max_replans": 1,               # 现状基线：生产默认值（Phase 1 才调到 3）
    "deadline_minutes": 10,
    "approval_timeout_seconds": 5,  # 生产 600s，评测等不起；timeout case 靠它快速跳过
}

# 默认桩返回（case.stub 可按工具名覆盖；必须像真话，占位假数据会被模型识破
# 并拒绝二次利用——2026-08-06 eval_golden 实锤的 sabotage 教训）
_STUB_RETURNS = {
    "web_search": "（桩）搜索结果：1. 微博热搜：某品牌发布新一代折叠屏手机，起售价 8999 元。2. 八部门印发人工智能产业高质量发展行动方案。3. 国家天文台发布系外行星观测成果。",
    "deep_research": "（桩）调研完成：报告已整理并发送，含三个核心要点与数据来源。",
    "get_time": "2026-08-12 15:30 星期三",
    "get_weather": "北京 明天 晴，25~33℃，微风。",
    "set_reminder": "提醒已设置成功，到点会叫你。",
    "query_chat_history": "（桩）近期聊天：讨论了 AI 平台选型、小主机装机、周末安排。",
    "bilibili_summary": "（桩）视频内容摘要：该讲座讲解了主题的三个要点：背景、现状、展望……",
    "watch_video": "（桩）已仔细观看：视频讲解了三个要点……观后感：内容扎实。",
    "ai_draw": "（桩）图片已生成并发送给对方。",
    "unified_tts": "（桩）语音已合成并发送。",
    "send_feed": "（桩）说说已发布。",
    "send_message": "（桩）消息已发送。",
}
_DEFAULT_STUB_RETURN = "（桩）操作成功。"


class _Usage:
    """token 成本累计（逐调用从响应 metadata 提取）。"""

    def __init__(self):
        self.calls = []  # [{model, input, output}]

    def record(self, model_name: str, resp) -> None:
        try:
            usage = getattr(resp, "response_metadata", {}).get("token_usage", {})
            self.calls.append({
                "model": model_name,
                "input": int(usage.get("prompt_tokens", 0)),
                "output": int(usage.get("completion_tokens", 0)),
            })
        except Exception:
            pass

    def totals(self) -> dict:
        return {"calls": len(self.calls),
                "input": sum(c["input"] for c in self.calls),
                "output": sum(c["output"] for c in self.calls)}


def _patch_models(usage: _Usage):
    """包装 junjun_llm.get_chat_model：每次 ainvoke 记录 token 用量。

    planner/executor 都在函数内部 from junjun_llm import get_chat_model，
    调用时取模块属性——补丁在源头模块即全局生效。
    """
    import junjun_llm
    real = junjun_llm.get_chat_model

    class _Wrap:
        def __init__(self, inner, name):
            self._inner, self._name = inner, name

        async def ainvoke(self, *a, **kw):
            resp = await self._inner.ainvoke(*a, **kw)
            usage.record(self._name, resp)
            return resp

        def __getattr__(self, k):
            return getattr(self._inner, k)

    def patched(task: str, *a, **kw):
        return _Wrap(real(task, *a, **kw), task)

    junjun_llm.get_chat_model = patched


def _make_stub_tools(called: list, stub_overrides: dict, fail_counters: dict):
    """真实工具 schema + 记录桩执行体。fail_counters: tool -> 剩余失败次数。"""
    from langchain_core.tools import StructuredTool
    from junjun_skills.registry import get_tools as real_get_tools

    stubs = []
    for t in real_get_tools():
        name = t.name

        async def _stub(_name=name, **kwargs):
            called.append((_name, kwargs))
            if fail_counters.get(_name, 0) > 0:
                fail_counters[_name] -= 1
                return "[TOOL_ERROR kind=timeout] 评测注入的模拟失败"
            ov = stub_overrides.get(_name)
            if isinstance(ov, str):
                return ov
            return _STUB_RETURNS.get(_name, _DEFAULT_STUB_RETURN)

        stubs.append(StructuredTool(
            name=name,
            description=t.description or "",
            args_schema=getattr(t, "args_schema", None),
            coroutine=_stub,
        ))
    return stubs


def _split_case_stub(case: dict) -> tuple:
    """case.stub -> (文本覆盖 dict, 失败计数 dict)。"""
    overrides, fails = {}, {}
    for tool, spec in (case.get("stub") or {}).items():
        if isinstance(spec, str):
            overrides[tool] = spec
        elif isinstance(spec, dict):
            if spec.get("fail_times"):
                fails[tool] = int(spec["fail_times"])
            if isinstance(spec.get("then"), str):
                overrides[tool] = spec["then"]
    return overrides, fails


async def _run_negative(kernel, case: dict) -> dict:
    """负例两层判定（与生产同路径）：
    1. route_to_task 判对话通道 -> 直接 PASS（0 token，绝大多数负例应死在这层）
    2. 路由误伤放行 -> try_submit 必须返回 None（内核是第二道防线）。
    """
    from junjun_agent.router import route_to_task
    if not route_to_task(case["input"], chat_id=f"eval:{case['id']}:private"):
        return {"id": case["id"], "pass": True}
    try:
        ack = await asyncio.wait_for(
            kernel.try_submit(case["input"], chat_id=f"eval:{case['id']}:private"),
            timeout=120)
    except asyncio.TimeoutError:
        return {"id": case["id"], "pass": False, "reason": "TIMEOUT(120s) 规划调用"}
    except Exception as e:
        return {"id": case["id"], "pass": False,
                "reason": f"ERROR {type(e).__name__}: {e}"}
    if ack is None:
        return {"id": case["id"], "pass": True,
                "reason": "（路由误伤放行，内核拒收兜底成功）"}
    return {"id": case["id"], "pass": False,
            "reason": f"闲聊被接单（路由+内核双层失守）: {ack[:40]}"}


async def _await_approval(runner, decision: bool, timeout=90) -> str:
    """轮询待审批队列并扮演管理员决议；超时返回错误串。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pend = list(runner.pending_approvals)
        if pend:
            await runner.resume(pend[0], decision)
            return ""
        await asyncio.sleep(0.5)
    return f"等待审批挂起超时（{timeout}s）"


def _judge_prompt(goal: str, artifact: str) -> str:
    return (f"给下面的任务产物质量打分（1-5，只回一个数字）。\n"
            f"任务目标：{goal}\n产物：\n{artifact[:2000]}\n"
            f"3 分 = 基本可用；4 分 = 结构清晰内容贴合；5 分 = 超出预期。")


async def _run_positive(kernel, runner, usage: _Usage, called: list, case: dict) -> dict:
    exp = case.get("expect", {})
    chat_id = f"eval:{case['id']}:private"
    completed = {}

    # 捕获终态 plan（report 节点与图异常路径都经 kernel._report）。
    # 注意：try_submit 只派后台任务，_report 在后面才发生——补丁必须活到终态，
    # 本函数结束才恢复（case 串行跑，无串扰）。
    orig_report = kernel._report

    async def _capture(plan):
        completed[plan.plan_id] = plan
        await orig_report(plan)

    kernel._report = _capture
    try:
        # 路由覆盖率观测（0 token，仅记录不判分）：route_to_task 宁漏勿错，
        # 大量真实委托到不了内核由对话通道兜底——这是独立维度（2026-08-12
        # 基线前测：18 条正例只放行 4 条），混进内核成功率会两头都看不清。
        from junjun_agent.router import route_to_task
        routed = route_to_task(case["input"], chat_id=chat_id)
        try:
            ack = await asyncio.wait_for(
                kernel.try_submit(case["input"], chat_id=chat_id),
                timeout=120)
        except asyncio.TimeoutError:
            return {"id": case["id"], "pass": False, "reason": "TIMEOUT(120s) 规划调用"}
        except Exception as e:
            return {"id": case["id"], "pass": False,
                    "reason": f"ERROR {type(e).__name__}: {e}"}
        if ack is None:
            return {"id": case["id"], "pass": False,
                    "reason": "规划返回 None（正例应接单）"}

        # 人审模拟
        approval = case.get("approval")
        approval_task = None
        if approval in ("approve", "reject"):
            approval_task = asyncio.create_task(
                _await_approval(runner, approval == "approve"))

        # 等终态（审批 timeout 由看门狗 5s 自动跳过）
        plan = None
        deadline = time.time() + 300
        while time.time() < deadline:
            if completed:
                plan = next(iter(completed.values()))
                break
            await asyncio.sleep(1.0)
        if approval_task is not None:
            approval_err = await approval_task
            if approval_err:
                return {"id": case["id"], "pass": False, "reason": approval_err}
        if plan is None:
            return {"id": case["id"], "pass": False, "reason": "TIMEOUT(300s) 未等到终态"}
    finally:
        kernel._report = orig_report

    fails = []
    steps = plan.steps
    called_names = [n for n, _ in called]
    if exp.get("plan_done") and plan.state != "done":
        fails.append(f"任务未完成（state={plan.state}，note={plan.note[:60]}，"
                     f"步骤: {[(s.id, s.status) for s in steps]}）")
    for spec in exp.get("must_use", []):
        alts = spec.split("|")
        if not any(a in called_names for a in alts):
            fails.append(f"未使用 {'/'.join(alts)}（实际调用: {called_names or '无'}）")
    for name in exp.get("must_not_use", []):
        if name in called_names:
            fails.append(f"不应使用 {name}")
    order = exp.get("order", [])
    if order:
        idx = [called_names.index(o) if o in called_names else -1 for o in order]
        if -1 in idx or idx != sorted(idx):
            fails.append(f"调用顺序不符（期望 {order}，实际 {called_names}）")
    if exp.get("min_steps") and len(steps) < exp["min_steps"]:
        fails.append(f"步数 {len(steps)} < 期望下限 {exp['min_steps']}")
    if exp.get("max_steps") and len(steps) > exp["max_steps"]:
        fails.append(f"步数 {len(steps)} > 期望上限 {exp['max_steps']}")
    if exp.get("min_replans") and plan.replans < exp["min_replans"]:
        fails.append(f"未发生重规划（replans={plan.replans}，期望 ≥{exp['min_replans']}）")
    for action, want in (exp.get("step_action_status") or {}).items():
        got = [s.status for s in steps if s.action == action]
        if not got:
            fails.append(f"最终计划里没有 {action} 步骤")
        elif want not in got:
            fails.append(f"{action} 步骤状态 {got}，期望含 {want}")
    haystack = "\n".join(s.result for s in steps if s.result)
    for s in exp.get("result_contains", []):
        if s not in haystack:
            fails.append(f"产物缺少「{s}」")

    # 质量抽检（产物类 case 才跑，控制成本）
    judge_score = None
    if exp.get("judge"):
        artifact = max((s.result for s in steps if s.status == "done" and s.result),
                       key=len, default="")
        if artifact:
            import junjun_llm
            from langchain_core.messages import HumanMessage
            model = junjun_llm.get_chat_model("thinker")
            resp = await model.ainvoke([HumanMessage(
                content=_judge_prompt(plan.goal, artifact))])
            usage.record("judge:thinker", resp)
            digits = "".join(c for c in str(resp.content)[:3] if c.isdigit())
            judge_score = int(digits[:1]) if digits else 0
            if judge_score < 3:
                fails.append(f"质量抽检 {judge_score} 分（<3）: {artifact[:60]}")

    return {"id": case["id"], "pass": not fails,
            "reason": "；".join(fails), "routed": routed,
            "state": plan.state, "steps": len(steps), "replans": plan.replans,
            "tools": called_names, "judge": judge_score,
            "step_detail": [{"id": s.id, "action": s.action, "status": s.status}
                            for s in steps]}


async def _main(args) -> int:
    cases = [json.loads(l) for l in
             CASES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    skipped = [c["id"] for c in cases if not c.get("enabled", True)]
    cases = [c for c in cases if c.get("enabled", True)]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if not cases:
        print("没有匹配的 case")
        return 1
    if skipped and not args.only:
        print(f"跳过占位 case（enabled:false）: {skipped}")

    from junjun_skills.registry import load_builtin
    from junjun_skills.plugin_loader import load_plugins
    load_builtin()
    load_plugins()

    # ---- 补丁面 ----
    usage = _Usage()
    _patch_models(usage)

    import junjun_skills.registry as reg
    called = []
    stubs_holder = {"tools": []}

    def _get_tools_patched(*a, **kw):
        return list(stubs_holder["tools"])

    reg.get_tools = _get_tools_patched
    reg.warm_tool_embeddings = lambda *a, **kw: asyncio.sleep(0)

    import junjun_agent.task_kernel.executor as executor
    executor._cfg = lambda: dict(_EVAL_CFG)
    from junjun_agent.task_kernel import graph as tk_graph
    # runner 不 configure：_persist_dir 保持 None -> 本意是内存 checkpointer，
    # 但 _ensure_graph 会开 aiosqlite ":memory:" 连接——其 worker 是非守护线程，
    # 脚本退出时解释器 _shutdown join 它 = 永远挂死（2026-08-12 py-spy 实锤：
    # case 37s 跑完写盘，MainThread 挂 threading._shutdown 17 分钟）。
    # 生产进程长跑无感，评测脚本短命必须换 MemorySaver（纯 dict 无线程；
    # interrupt/resume 语义一致，build_graph docstring 的测试先例）。
    from langgraph.checkpoint.memory import MemorySaver
    tk_graph.runner._graph = tk_graph.build_graph(MemorySaver())

    # 副作用封堵：审批通知/主动发送/结局登记
    import junjun_core.security as sec
    sec.notify_admin = lambda *a, **kw: asyncio.sleep(0, result=True)
    import junjun_agent.outbound as outbound

    async def _fake_send(chat_id_, segments, **kw):
        return True

    outbound.send_proactive = _fake_send
    from junjun_agent.tasks import task_manager
    task_manager._record_outcome = lambda *a, **kw: None

    results = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        overrides, fail_counters = _split_case_stub(case)
        called.clear()
        stubs_holder["tools"] = _make_stub_tools(called, overrides, fail_counters)

        if case.get("expect", {}).get("submit_rejected"):
            r = await _run_negative(executor.kernel, case)
        else:
            r = await _run_positive(executor.kernel, tk_graph.runner, usage,
                                    called, case)
        results.append(r)
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{i}/{len(cases)}] {mark} {r['id']}"
              + ("" if r["pass"] else f"  -- {r['reason']}"))
        _write_report(results, t0, usage)

    passed = sum(1 for r in results if r["pass"])
    steps_list = [r["steps"] for r in results if r.get("steps")]
    routed_hits = sum(1 for r in results if r.get("routed"))
    routed_base = sum(1 for r in results if "routed" in r)
    print(f"\n==== {passed}/{len(results)} 通过，耗时 {time.time()-t0:.0f}s ====")
    if steps_list:
        print(f"指标：成功率 {passed}/{len(results)}"
              f" | 平均步数 {sum(steps_list)/len(steps_list):.1f}"
              f" | 路由覆盖率 {routed_hits}/{routed_base}")
    print(f"token：{usage.totals()}")
    print(f"报告: {_write_report(results, t0, usage)}")
    if args.baseline:
        src = REPORT_DIR / "eval_tasks_report_latest.json"
        BASELINE_FILE.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"基线存档: {BASELINE_FILE}")
    return 0 if passed == len(results) else 1


def _write_report(results: list, t0: float, usage: _Usage) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t0))
    path = REPORT_DIR / f"eval_tasks_report_{ts}.json"
    passed = sum(1 for r in results if r["pass"])
    steps_list = [r["steps"] for r in results if r.get("steps")]
    routed_hits = sum(1 for r in results if r.get("routed"))
    routed_base = sum(1 for r in results if "routed" in r)
    path.write_text(json.dumps({
        "ts": ts, "passed": passed, "total": len(results),
        "success_rate": passed / len(results) if results else 0,
        "avg_steps": (sum(steps_list) / len(steps_list)) if steps_list else 0,
        "router_coverage": {"routed": routed_hits, "total": routed_base},
        "tokens": usage.totals(), "token_calls": usage.calls,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "eval_tasks_report_latest.json").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _show_latest() -> int:
    p = REPORT_DIR / "eval_tasks_report_latest.json"
    if not p.exists():
        print("还没有任务评测报告")
        return 1
    rep = json.loads(p.read_text(encoding="utf-8"))
    print(f"报告 {rep['ts']}: {rep['passed']}/{rep['total']}"
          f" | 平均步数 {rep.get('avg_steps', 0):.1f} | token {rep.get('tokens')}")
    for r in rep["results"]:
        if not r["pass"]:
            print(f"  FAIL {r['id']} -- {r['reason']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑 id 包含该子串的 case")
    ap.add_argument("--report", action="store_true", help="只看最近一次报告")
    ap.add_argument("--baseline", action="store_true", help="跑完另存基线存档")
    args = ap.parse_args()
    if args.report:
        sys.exit(_show_latest())
    sys.exit(asyncio.run(_main(args)))
