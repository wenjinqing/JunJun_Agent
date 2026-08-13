"""一次性排查：从 Langfuse 找 HonestyGuard 误拦的那条原文。

扫描最近 N 小时 agent.* span 的 output.reply，用 honesty_guard 的
（有 bug 的）声称模式逐个匹配——命中的就是会被拦截/替换的回复。
结果写 data/honesty_scan_<ts>.json，控制台只打 ascii 摘要。
"""

import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

# 系统代理（Windows 注册表）会拦截 localhost 请求——显式空代理绕开
_opener = build_opener(ProxyHandler({}))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

sys.path.insert(0, str(ROOT))
from junjun_agent.honesty_guard import _CLAIM_RULES

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 48


def _auth() -> str:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip('"')
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip('"')
    return "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()


def main():
    host = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").strip('"')
    auth = _auth()
    since = datetime.now(timezone.utc) - timedelta(hours=HOURS)
    obs = []
    for page in range(1, 11):
        qs = (f"page={page}&limit=100"
              f"&fromStartTime={since.strftime('%Y-%m-%dT%H:%M:%S.000Z')}")
        req = Request(f"{host}/api/public/observations?{qs}",
                      headers={"Authorization": auth})
        with _opener.open(req, timeout=15) as r:
            items = json.loads(r.read()).get("data", [])
        obs.extend(items)
        if len(items) < 100:
            break

    hits = []
    scanned = 0
    for o in obs:
        name = o.get("name") or ""
        if not name.startswith("agent."):
            continue
        out = o.get("output") or {}
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                continue
        reply = (out.get("reply") or "") if isinstance(out, dict) else ""
        if not reply:
            continue
        scanned += 1
        matched = []
        for rule in _CLAIM_RULES:
            for pat in rule["patterns"]:
                m = re.search(pat, reply)
                if m:
                    matched.append({"rule": rule["name"], "pattern": pat,
                                    "hit": m.group(0)})
                    break
        if matched:
            hits.append({
                "trace_id": o.get("traceId"),
                "start": o.get("startTime"),
                "span": name,
                "matched": matched,
                "reply": reply[:300],
            })

    ts = time.strftime("%Y%m%d_%H%M%S")
    report = ROOT / "data" / f"honesty_scan_{ts}.json"
    report.write_text(json.dumps(hits, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"scanned={scanned} agent replies, pattern-hits={len(hits)}")
    for h in hits:
        rules = ",".join(f"{m['rule']}:{m['hit']}" for m in h["matched"])
        print(f"  {h['start']} trace={h['trace_id']}")
        print(f"    rules(escaped): {rules.encode('unicode_escape').decode()[:120]}")
        print(f"    reply(escaped): {h['reply'][:150].encode('unicode_escape').decode()}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
