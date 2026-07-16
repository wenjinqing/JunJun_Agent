"""WebUI 后端：FastAPI（配置/日志/统计/数据管理）。

与原 src/webui 的偏离（2026-07-16 决策）：不复用旧前端 dist——旧前端与旧 API
深度耦合且一半路由服务于已砍掉的功能（多 bot 管理/git 镜像）。改为轻量单页
（static/index.html）+ 精简 API，覆盖阶段 7 验收项：配置热改/日志实时/统计/管理。

鉴权：WEBUI_TOKEN 环境变量（Bearer）；未设置时仅允许 127.0.0.1。
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from junjun_core.observability import get_logger

logger = get_logger("webui")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------- 日志环形缓冲 + 广播 ----------

_LOG_BUFFER: deque = deque(maxlen=500)
_WS_CLIENTS: set = set()
_MAX_WS_CLIENTS = 10


class _BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            line = self.format(record)
        except Exception:
            return
        _LOG_BUFFER.append(line)
        for ws in list(_WS_CLIENTS):
            try:
                asyncio.get_event_loop().create_task(ws.send_text(line))
            except Exception:
                _WS_CLIENTS.discard(ws)


def install_log_capture() -> None:
    h = _BufferHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s", "%H:%M:%S"))
    h.setLevel(logging.INFO)
    logging.root.addHandler(h)


# ---------- 鉴权 ----------

def _check_auth(request: Request) -> None:
    token = os.environ.get("WEBUI_TOKEN", "")
    if token:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            raise HTTPException(401, "invalid token")
    else:
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "local access only (set WEBUI_TOKEN for remote)")


app = FastAPI(title="JunJun WebUI", docs_url=None, redoc_url=None)

# ---------- 配置 ----------

# 允许热改的白名单（section, key）——防 WebUI 改崩关键结构
_MUTABLE_KEYS = {
    ("chat", "talk_value"), ("chat", "mentioned_bot_reply"), ("chat", "max_context_size"),
    ("chat", "enable_talk_value_rules"),
    ("mood", "enable_mood"), ("emoji", "emoji_chance"), ("emoji", "steal_emoji"),
    ("repeat", "enable"), ("repeat", "threshold"),
    ("proactive_chat", "enable"), ("proactive_chat", "max_daily_proactive"),
    ("response_splitter", "enable"), ("chinese_typo", "enable"),
    ("personality", "personality"), ("personality", "reply_style"), ("personality", "interest"),
}


@app.get("/api/config", dependencies=[Depends(_check_auth)])
def get_config():
    from junjun_core.config import get_global_config
    raw = get_global_config().raw
    out = {}
    for section, key in _MUTABLE_KEYS:
        out.setdefault(section, {})[key] = raw.get(section, {}).get(key)
    return out


@app.post("/api/config", dependencies=[Depends(_check_auth)])
async def set_config(payload: dict):
    """热改内存配置（白名单键）。持久化到 toml 由用户手动确认（防误触写坏）。"""
    from junjun_core.config import get_global_config
    raw = get_global_config().raw
    changed = []
    for section, kv in payload.items():
        if not isinstance(kv, dict):
            continue
        for key, value in kv.items():
            if (section, key) not in _MUTABLE_KEYS:
                raise HTTPException(400, f"不可修改的配置项: {section}.{key}")
            raw.setdefault(section, {})[key] = value
            changed.append(f"{section}.{key}={value}")
    if changed:
        logger.info(f"WebUI 配置热改: {', '.join(changed)}")
    return {"changed": changed}


# ---------- 日志 ----------

@app.get("/api/logs", dependencies=[Depends(_check_auth)])
def get_logs(limit: int = 200):
    return {"lines": list(_LOG_BUFFER)[-limit:]}


@app.websocket("/api/logs/ws")
async def logs_ws(ws: WebSocket):
    token = os.environ.get("WEBUI_TOKEN", "")
    if token and ws.query_params.get("token") != token:
        await ws.close(code=4401)
        return
    if len(_WS_CLIENTS) >= _MAX_WS_CLIENTS:
        await ws.close(code=4429)
        return
    await ws.accept()
    _WS_CLIENTS.add(ws)
    try:
        for line in list(_LOG_BUFFER)[-50:]:
            await ws.send_text(line)
        while True:
            # 心跳（客户端 ping 或 30s 超时断开）
            await asyncio.wait_for(ws.receive_text(), timeout=60)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        _WS_CLIENTS.discard(ws)


# ---------- 统计 ----------

@app.get("/api/stats", dependencies=[Depends(_check_auth)])
def get_stats(hours: int = 24):
    from junjun_core.database import Messages, LLMUsage
    from peewee import fn
    since = time.time() - hours * 3600
    total = Messages.select().where(Messages.time >= since).count()
    replied = Messages.select().where(
        (Messages.time >= since) & (Messages.is_bot == True)).count()  # noqa: E712
    usage = [{
        "type": u.request_type or "?",
        "calls": int(u.n), "prompt_tokens": int(u.pt or 0), "completion_tokens": int(u.ct or 0),
    } for u in (LLMUsage
                .select(LLMUsage.request_type,
                        fn.COUNT(LLMUsage.id).alias("n"),
                        fn.SUM(LLMUsage.prompt_tokens).alias("pt"),
                        fn.SUM(LLMUsage.completion_tokens).alias("ct"))
                .where(LLMUsage.time >= since).group_by(LLMUsage.request_type))]
    return {"received": total - replied, "replied": replied, "usage": usage}


# ---------- 会话 ----------

@app.get("/api/sessions", dependencies=[Depends(_check_auth)])
def get_sessions():
    from junjun_core.gateway.session_manager import get_session_manager
    out = []
    for s in get_session_manager().all_sessions().values():
        out.append({
            "chat_id": s.chat_id, "is_group": s.is_group,
            "silenced": s.silenced_until_call,
            "last_active": s.last_active_ts,
            "context_len": len(s.memory.entries) if s.memory else 0,
        })
    return out


# ---------- 数据管理（黑话/表达/表情/提醒/画像查看与删除）----------

@app.get("/api/jargon", dependencies=[Depends(_check_auth)])
def list_jargon():
    from junjun_core.database import Jargon
    return [{"id": r.id, "term": r.term, "explanation": r.explanation, "count": r.count}
            for r in Jargon.select().order_by(Jargon.count.desc()).limit(200)]


@app.delete("/api/jargon/{row_id}", dependencies=[Depends(_check_auth)])
def delete_jargon(row_id: int):
    from junjun_core.database import Jargon
    n = Jargon.delete().where(Jargon.id == row_id).execute()
    return {"deleted": n}


@app.get("/api/expressions", dependencies=[Depends(_check_auth)])
def list_expressions():
    from junjun_core.database import Expression
    return [{"id": r.id, "chat_id": r.chat_id, "situation": r.situation,
             "style": r.style, "count": r.count}
            for r in Expression.select().order_by(Expression.count.desc()).limit(200)]


@app.delete("/api/expressions/{row_id}", dependencies=[Depends(_check_auth)])
def delete_expression(row_id: int):
    from junjun_core.database import Expression
    return {"deleted": Expression.delete().where(Expression.id == row_id).execute()}


@app.get("/api/reminders", dependencies=[Depends(_check_auth)])
def list_reminders_api():
    from junjun_core.database import ReminderTasks
    return [{"task_id": r.task_id, "chat_id": r.chat_id, "content": r.content,
             "remind_time": r.remind_time, "repeat": r.repeat_type}
            for r in ReminderTasks.select().where(
                (ReminderTasks.is_completed == False)          # noqa: E712
                & (ReminderTasks.is_cancelled == False))]      # noqa: E712


@app.get("/api/persons", dependencies=[Depends(_check_auth)])
def list_persons():
    from junjun_core.database import PersonInfo
    out = []
    for r in PersonInfo.select().limit(200):
        try:
            points = json.loads(r.memory_points or "[]")
        except json.JSONDecodeError:
            points = []
        out.append({"user_id": r.user_id, "name": r.person_name, "points": points})
    return out


# ---------- 静态页 ----------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))


async def start_webui() -> Optional[asyncio.Task]:
    """与网关同进程启动（WEBUI_ENABLED=true 时）。返回 server task。"""
    if os.environ.get("WEBUI_ENABLED", "false").lower() != "true":
        logger.info("WebUI 未启用（WEBUI_ENABLED != true）")
        return None
    import uvicorn
    host = os.environ.get("WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("WEBUI_PORT", "8002"))
    install_log_capture()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name="webui")
    logger.info(f"WebUI 已启动 http://{host}:{port}"
                + ("" if os.environ.get("WEBUI_TOKEN") else "（未设 WEBUI_TOKEN，仅限本机访问）"))
    return task
