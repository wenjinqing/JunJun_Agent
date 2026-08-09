"""TaskKernel 灰度抽检（架构重写方案阶段 4 观察期工具）。

从 Langfuse 拉 router.* / task_kernel.* span，输出：
- 健康度面板：接单数 / 完成率 / replan 率 / 验证失败 / 耗时分布
- 可疑清单：规划失败回退（accepted=false）、失败计划、超长任务——
  误路由证据主要靠 accepted=false 与失败 goal 的人工判读

用法：
    uv run python scripts/audit_task_kernel.py            # 最近 3 天
    uv run python scripts/audit_task_kernel.py --days 7
报告同时落盘 data/audit_taskkernel_<ts>.json。
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPORT_DIR = ROOT / "data"
_PAGE_LIMIT = 100
_MAX_PAGES = 10


def _auth_header() -> str:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip('"')
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip('"')
    if not (pk and sk):
        sys.exit("LANGFUSE_PUBLIC_KEY/SECRET_KEY 未配置（.env）")
    return "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()


def _fetch_observations(host: str, auth: str, since: datetime) -> tuple:
    """拉取时间窗内全部 observation（分页），客户端再按名字过滤。

    返回 (items, truncated)。打满分页上限时 truncated=True——活跃 bot 每条
    消息多个 observation，3 天窗口轻松破千；静默截断会让健康度面板只反映
    窗口的一部分（最老样本被丢）还宣称全覆盖（2026-08-06 审查实锤）。
    """
    out = []
    truncated = False
    for page in range(1, _MAX_PAGES + 1):
        qs = (f"page={page}&limit={_PAGE_LIMIT}"
              f"&fromStartTime={since.strftime('%Y-%m-%dT%H:%M:%S.000Z')}")
        req = Request(f"{host}/api/public/observations?{qs}",
                      headers={"Authorization": auth})
        with urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        items = data.get("data", [])
        out.extend(items)
        if len(items) < _PAGE_LIMIT:
            break
    else:
        truncated = True
    return out, truncated


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3, help="回看天数（默认 3）")
    args = ap.parse_args()

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    obs, truncated = _fetch_observations(host, _auth_header(), since)

    routers = [o for o in obs if (o.get("name") or "").startswith("router.")]
    kernels = [o for o in obs
               if (o.get("name") or "").startswith("task_kernel.")
               and not (o.get("name") or "").startswith("task_kernel_step.")]
    steps = [o for o in obs if (o.get("name") or "").startswith("task_kernel_step.")]

    def md(o):
        return o.get("metadata") or {}

    # ---------- 健康度面板 ----------
    accepted = [o for o in routers if md(o).get("accepted") is True]
    fallback = [o for o in routers if md(o).get("accepted") is False]
    done = [o for o in kernels if md(o).get("state") == "done"]
    failed = [o for o in kernels if md(o).get("state") == "failed"]
    replanned = [o for o in kernels if int(md(o).get("replans") or 0) > 0]
    verify_fails = sum(int(md(o).get("verify_failures") or 0) for o in kernels)
    durations = sorted(float(md(o).get("duration_s") or 0) for o in kernels)
    step_failed = [o for o in steps if md(o).get("status") == "failed"]

    n = len(kernels)
    print(f"== TaskKernel 抽检（{since.date()} 起 {args.days} 天，span 共 {len(obs)} 个）==")
    if truncated:
        print(f"!! 警告：样本达到 {_MAX_PAGES * _PAGE_LIMIT} 条分页上限，最老样本被截断——"
              "以下面板只反映窗口的一部分，缩短 --days 再跑")
    print(f"路由命中: {len(routers)}  接单: {len(accepted)}  规划失败回退: {len(fallback)}")
    # 回退原因分布（disabled=灰度开关关着，不是缺陷；planner_none/exception 才是）
    if fallback:
        from collections import Counter
        reasons = Counter(str(md(o).get("reject_reason") or "未知(旧数据)")
                          for o in fallback)
        print("回退原因: " + "  ".join(f"{k}={v}" for k, v in reasons.most_common()))
    if n:
        print(f"计划: {n}  完成: {len(done)}  失败: {len(failed)}"
              f"  完成率: {len(done)/n:.0%}  replan 率: {len(replanned)/n:.0%}")
        print(f"验证失败(步次): {verify_fails}  步骤失败(步次): {len(step_failed)}")
        print(f"耗时(s): 中位 {durations[n//2]:.0f}  最大 {durations[-1]:.0f}")
    else:
        print("计划: 0（观察窗内无复杂任务样本）")

    # ---------- 可疑清单（人工判读素材） ----------
    suspects = []
    for o in fallback:
        # 开关关闭期的样本不是可疑项——灰度来回切的日子里它们全是噪声
        if str(md(o).get("reject_reason") or "") == "disabled":
            continue
        text = str((o.get("input") or {}).get("latest_text", ""))[:50]
        suspects.append({"kind": "规划失败回退", "text": text,
                         "reason": str(md(o).get("reject_reason") or ""),
                         "time": o.get("startTime")})
    for o in failed:
        m = md(o)
        suspects.append({"kind": "计划失败",
                         "text": str((o.get("input") or {}).get("goal", ""))[:50],
                         "note": str(m.get("note", ""))[:80],
                         "time": o.get("startTime")})
    for o in kernels:
        if float(md(o).get("duration_s") or 0) > 1200:
            suspects.append({"kind": "超长任务(>20min)",
                             "text": str((o.get("input") or {}).get("goal", ""))[:50],
                             "time": o.get("startTime")})
    if suspects:
        print("\n-- 可疑样本（逐条人工判读是不是误路由/真缺陷）--")
        for s in suspects:
            print(f"  [{s['kind']}] {s['text']}" + (f"  -- {s.get('note')}" if s.get("note") else ""))

    REPORT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"audit_taskkernel_{ts}.json"
    path.write_text(json.dumps({
        "ts": ts, "days": args.days, "truncated": truncated,
        "router_hits": len(routers), "accepted": len(accepted), "fallback": len(fallback),
        "plans": n, "done": len(done), "failed": len(failed),
        "replanned": len(replanned), "verify_failures": verify_fails,
        "suspects": suspects,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
