# -*- coding: utf-8 -*-
"""一次性：按 trace id 拉 Langfuse trace 全文，落 data/trace_view.json。"""
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _env(name):
    v = os.environ.get(name, "").strip()
    if v:
        return v
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

def main(trace_id):
    import httpx
    host = _env("LANGFUSE_HOST").rstrip("/") or _env("LANGFUSE_BASE_URL").rstrip("/")
    pk, sk = _env("LANGFUSE_PUBLIC_KEY"), _env("LANGFUSE_SECRET_KEY")
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    with httpx.Client(headers={"Authorization": f"Basic {auth}"}, timeout=30.0,
                      trust_env=False) as c:  # 系统代理会拦 localhost（502）
        r = c.get(f"{host}/api/public/traces/{trace_id}")
        r.raise_for_status()
        t = r.json()
    out = {
        "id": t.get("id"), "name": t.get("name"), "timestamp": t.get("timestamp"),
        "latency": t.get("latency"), "tags": t.get("tags"),
        "input": t.get("input"), "output": t.get("output"),
        "metadata": t.get("metadata"),
        "observations": [
            {
                "type": o.get("type"), "name": o.get("name"),
                "start": o.get("startTime"), "end": o.get("endTime"),
                "level": o.get("level"), "status": o.get("statusMessage"),
                "model": o.get("model"),
                "usage": o.get("usage") or o.get("usageDetails"),
                "input": o.get("input"), "output": o.get("output"),
            }
            for o in sorted(t.get("observations") or [],
                            key=lambda x: x.get("startTime") or "")
        ],
    }
    p = ROOT / "data" / f"trace_{trace_id[:8]}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"obs={len(out['observations'])} -> {p}")

if __name__ == "__main__":
    main(sys.argv[1])
