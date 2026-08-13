"""run_code 沙箱 HTTP 服务：POST /run -> 一次性 docker 容器执行（TaskKernel Phase 2）。

定位：编排层，不是安全边界。真正的隔离全部在容器：
- 每跑一次起一个全新容器，--rm 自动销毁，无状态残留
- --network=none        容器无网络（出入都没有）
- --read-only + tmpfs   根文件系统只读，仅 /tmp 临时可写（noexec）
- --memory=2g --cpus=2  资源限额
- --user sandbox        非 root（镜像内 uid 10001）
- 唯一挂载              data/workspace/<会话子目录> -> /workspace

逃逸面三清单（挂载/网络/用户）见 docs/沙箱逃逸面清单_2026-08-13.md。

启动：uv run uvicorn sandbox.server:app --host 127.0.0.1 --port 8100
前置：docker daemon 运行中 + 镜像已构建（见 sandbox/README.md）
"""

import asyncio
import os
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("SANDBOX_WORKSPACE_ROOT", "data/workspace")).resolve()
IMAGE = os.environ.get("SANDBOX_IMAGE", "junjun-sandbox:latest")
_TOKEN = os.environ.get("SANDBOX_TOKEN", "")   # 共享鉴权；空 = 不强制（仅本机回环）
MAX_TIMEOUT = 30            # 代码执行硬上限（秒）
GRACE = 8                   # 容器启停宽限（秒）
MAX_CODE = 64 * 1024        # 代码体积上限
MAX_CAPTURE = 256 * 1024    # stdout/stderr 各自捕获上限（超限即杀：输出失控=异常）
MAX_FILES = 50
_SEM = asyncio.Semaphore(4)  # 并发运行上限（防批量起容器打爆宿主机）

app = FastAPI(title="junjun-sandbox", docs_url=None, redoc_url=None, openapi_url=None)


class RunReq(BaseModel):
    code: str = Field(max_length=MAX_CODE)
    timeout: int = MAX_TIMEOUT
    workdir: str = "default"


def _safe(s: str) -> str:
    """workdir -> 目录名片段；与插件侧 _safe_name 同一规则（清洗是幂等的）。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:64] or "default"


def _resolve_workdir(raw: str) -> Path:
    """workdir 必须解析到 ROOT 之内的子目录。2026-08-13 审查 P0：_safe 放行点号，
    ".." 原样穿过后 wd = data/——整个生产库和号池被挂进容器。resolve 后断言归属，
    "."（=ROOT 本身，跨会话全通）同样拒。"""
    wd = (ROOT / _safe(raw)).resolve()
    if wd == ROOT or ROOT not in wd.parents:
        raise ValueError(f"非法 workdir: {raw!r}")
    return wd


def _snapshot(wd: Path) -> dict:
    snap = {}
    for p in wd.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                snap[p.relative_to(wd).as_posix()] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return snap


def _changed(wd: Path, before: dict) -> list:
    out = []
    for p in sorted(wd.rglob("*")):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(wd).as_posix()
        if before.get(rel) != (st.st_mtime_ns, st.st_size):
            out.append({"path": rel, "size": st.st_size})
        if len(out) >= MAX_FILES:
            break
    return out


async def _pump(stream, buf: bytearray, state: dict) -> None:
    """读管道到 buf（封顶 MAX_CAPTURE，超限置 overflow 由主循环立即杀）。"""
    while True:
        chunk = await stream.read(16384)
        if not chunk:
            return
        room = MAX_CAPTURE - len(buf)
        if room > 0:
            buf.extend(chunk[:room])
        if len(chunk) > max(room, 0):
            state["overflow"] = True
            return  # 不再排空——主循环会杀掉容器，管道随进程死而 EOF


async def _docker_kill(name: str) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "kill", name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(proc.wait(), 10)
    except Exception:
        pass


@app.post("/run")
async def run(req: RunReq, x_sandbox_token: str = Header("")):
    # 可选共享 token：.env 设了 SANDBOX_TOKEN 就强制鉴权（本机多进程环境
    # 「绑回环就够」的假设太脆，2026-08-13 审查）；未设置 = 仅本机开发，放行。
    if _TOKEN and x_sandbox_token != _TOKEN:
        raise HTTPException(401, "bad sandbox token")
    try:
        wd = _resolve_workdir(req.workdir)
    except ValueError as e:
        raise HTTPException(400, str(e))
    timeout = max(1, min(int(req.timeout), MAX_TIMEOUT))
    wd.mkdir(parents=True, exist_ok=True)
    before = _snapshot(wd)
    name = f"jj-sandbox-{uuid.uuid4().hex[:8]}"
    # Windows docker CLI 接受正斜杠盘符路径；as_posix 统一
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network=none", "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
        # 只读根 fs 下 matplotlib 配置目录不可写会每次重建字体缓存并刷警告——指到 tmpfs
        "-e", "MPLCONFIGDIR=/tmp/mpl",
        "--memory=2g", "--cpus=2", "--user", "sandbox",
        "-v", f"{wd.as_posix()}:/workspace", "-w", "/workspace",
        IMAGE,
    ]
    started = time.monotonic()
    state = {"overflow": False}
    killed = False
    kill_reason = ""
    async with _SEM:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            return {"ok": False, "error": "docker CLI 不在 PATH（沙箱宿主配置缺失）"}
        proc.stdin.write(req.code.encode("utf-8"))
        try:
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 容器秒死（如镜像缺失）——错误看 stderr
        out_buf, err_buf = bytearray(), bytearray()
        pumps = [asyncio.create_task(_pump(proc.stdout, out_buf, state)),
                 asyncio.create_task(_pump(proc.stderr, err_buf, state))]
        waiter = asyncio.create_task(proc.wait())
        deadline = started + timeout + GRACE
        while True:
            done, _ = await asyncio.wait({waiter}, timeout=0.5)
            if waiter in done:
                break
            if state["overflow"]:
                killed, kill_reason = True, "输出超过 256KB 上限"
                break
            if time.monotonic() > deadline:
                killed, kill_reason = True, f"超过 {timeout}s 上限"
                break
        if killed:
            await _docker_kill(name)
            try:
                await asyncio.wait_for(waiter, 15)
            except asyncio.TimeoutError:
                proc.kill()
                await waiter
        await asyncio.gather(*pumps, return_exceptions=True)
    duration_ms = int((time.monotonic() - started) * 1000)
    stderr_text = err_buf.decode("utf-8", errors="replace")
    if kill_reason:
        stderr_text = (stderr_text + f"\n[sandbox] 已强制终止：{kill_reason}").strip()
    return {
        "ok": proc.returncode == 0 and not killed,
        "killed": killed,
        "returncode": proc.returncode,
        "duration_ms": duration_ms,
        "stdout": out_buf.decode("utf-8", errors="replace"),
        "stderr": stderr_text,
        "files": _changed(wd, before),
    }


async def _cmd_ok(*args) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(proc.wait(), 8)
        return proc.returncode == 0
    except Exception:
        return False


@app.get("/health")
async def health():
    docker_ok = await _cmd_ok("docker", "info", "--format", "ok")
    image_ok = docker_ok and await _cmd_ok("docker", "image", "inspect", IMAGE)
    return {"docker": docker_ok, "image": IMAGE if image_ok else None,
            "workspace_root": str(ROOT)}
