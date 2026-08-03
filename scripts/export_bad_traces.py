#!/usr/bin/env python
"""导出坏 case 候选（P8-1）：Langfuse trace -> 规则检测 -> JSONL 候选集。

用法：
    python scripts/export_bad_traces.py [--hours 24] [--limit 200] [--out PATH]

工作流：导出候选 -> 人工/LLM 标注挑真坏 case -> 固化进
tests/regression_corpus/cases.jsonl（pytest 回放，防检测器与功能双双退化）。

鉴权读 .env 的 LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY（只读 GET，不写）。
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from junjun_agent.loop.bad_cases import detect_bad_case  # noqa: E402


def _env(name: str) -> str:
    # 先读进程环境，再兜底解析 .env（脚本独立运行，不走 config 加载链）
    v = os.environ.get(name, "").strip()
    if v:
        return v
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def fetch_traces(hours: int, limit: int) -> list:
    import httpx
    host = _env("LANGFUSE_HOST").rstrip("/")
    pk, sk = _env("LANGFUSE_PUBLIC_KEY"), _env("LANGFUSE_SECRET_KEY")
    if not (host and pk and sk):
        raise SystemExit("缺少 LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY（.env 或环境变量）")
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600))
    traces, page = [], 1
    with httpx.Client(headers={"Authorization": f"Basic {auth}"}, timeout=30.0) as client:
        while len(traces) < limit:
            resp = client.get(f"{host}/api/public/traces",
                              params={"limit": min(100, limit - len(traces)),
                                      "page": page, "fromTimestamp": since})
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("data") or []
            if not batch:
                break
            traces.extend(batch)
            if len(batch) < 100:
                break
            page += 1
    return traces[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "data" / "bad_cases"
        / f"bad_cases_{time.strftime('%Y%m%d_%H%M')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    traces = fetch_traces(args.hours, args.limit)
    print(f"拉到 {len(traces)} 条 trace（近 {args.hours}h）")
    hits = 0
    with out.open("w", encoding="utf-8") as f:
        for t in traces:
            kinds = detect_bad_case(t)
            if not kinds:
                continue
            hits += 1
            f.write(json.dumps({
                "trace_id": t.get("id", ""),
                "kinds": kinds,
                "name": t.get("name", ""),
                "timestamp": t.get("timestamp", ""),
                "input_excerpt": str(t.get("input"))[:300],
                "output_excerpt": str(t.get("output"))[:300],
            }, ensure_ascii=False) + "\n")
    print(f"规则命中 {hits} 条候选 -> {out}")
    print("下一步：人工挑真坏 case 固化进 tests/regression_corpus/cases.jsonl")


if __name__ == "__main__":
    main()
