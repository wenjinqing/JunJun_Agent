"""workspace 插件 0-token 测试：门禁 / 静态预检 / 路径隔离 / SSRF / 内核人审放行。

Phase 2 安全验收（docs/Agent内核升级计划_TaskKernel_2026-08-12.md）：
「非管理员触发 run_code 被门禁拦下（误判回归同 commit）」——本文件即该回归。
全部不触网、不写生产库：沙箱 HTTP 用假 client，工作区根 monkeypatch 到 tmp。
"""

import json
from pathlib import Path

import pytest

from junjun_skills.plugins.workspace import tools as wt


@pytest.fixture()
def ws_root(tmp_path, monkeypatch):
    """工作区根指到 tmp；会话钉在私聊 qq:111。"""
    monkeypatch.setattr(wt, "_ROOT", tmp_path / "ws")
    from junjun_skills.builtin.memory_skills import current_chat_id
    token = current_chat_id.set("qq:111:private")
    yield tmp_path / "ws"
    current_chat_id.reset(token)


async def _call(tool, args: dict) -> str:
    """调用工具，异常折成文本——全量套件里 registry 会把工具包一层
    _wrap_error_feedback（异常 -> [TOOL_ERROR] 文本），单跑本文件时又是裸的，
    两种上下文都要兼容。"""
    try:
        return await tool.ainvoke(args)
    except Exception as e:
        return f"[raised] {type(e).__name__}: {e}"


# ---------------------------------------------------------------- 静态预检

class TestStaticScan:
    @pytest.mark.parametrize("code", [
        "import os\nprint(os.getcwd())",
        "import os, json",            # 逗号并列（正则挡不住，ast 必须抓到）
        "import json, os",            # 藏在第二个位置
        "from os import path",
        "from subprocess import run",
        "import socket",
        "import ctypes",
        "__import__('os')",
        "import sys\nsys.exit(0)",
        "import importlib",
    ])
    def test_blocked(self, code):
        assert wt._static_scan(code) != ""

    @pytest.mark.parametrize("code", [
        "import pandas as pd\nprint(pd.__version__)",
        "import json, math, pathlib",   # 白名单并列不得误伤
        "from pathlib import Path\nPath('a.txt').write_text('x')",
        "open('result.txt', 'w').write('hi')",
        "import matplotlib\nmatplotlib.use('Agg')",
        "from datetime import datetime",
        "print('os 字样出现在字符串里不该误伤')",
        "x = 'import os'  # 字符串字面量不是导入",
    ])
    def test_allowed(self, code):
        assert wt._static_scan(code) == ""

    def test_syntax_error_reported(self):
        assert "语法错误" in wt._static_scan("def f(:")


# ---------------------------------------------------------------- 路径隔离

class TestPathIsolation:
    @pytest.mark.parametrize("p", [
        "../outside.txt",
        "sub/../../outside.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "c:\\windows\\win.ini",
        "~/secret",
        "",
    ])
    def test_traversal_rejected(self, ws_root, p):
        with pytest.raises(ValueError):
            wt._resolve(p)

    def test_normal_paths_ok(self, ws_root):
        assert wt._resolve("a.txt").name == "a.txt"
        assert wt._resolve("sub/dir/b.md").parent.name == "dir"

    def test_chat_id_sanitized(self, ws_root):
        # chat_id 含冒号（qq:111:private）——Windows 文件名非法字符必须清洗掉
        d = wt._session_dir(create=True)
        assert ":" not in d.name and d.name == "qq_111_private"

    @pytest.mark.asyncio
    async def test_write_read_list_roundtrip(self, ws_root):
        r = await wt.workspace_write.ainvoke({"path": "notes/t.md", "content": "标题\n正文"})
        assert "已存到工作区" in r
        text = await wt.workspace_read.ainvoke({"path": "notes/t.md"})
        assert text == "标题\n正文"
        listing = await wt.workspace_list.ainvoke({})
        assert "notes/t.md" in listing and "2KB" not in listing  # 小文件按 B 显示

    @pytest.mark.asyncio
    async def test_read_missing_raises(self, ws_root):
        r = await _call(wt.workspace_read, {"path": "ghost.txt"})
        assert "没有" in r

    @pytest.mark.asyncio
    async def test_write_too_big(self, ws_root):
        r = await _call(wt.workspace_write,
                        {"path": "big.txt", "content": "x" * (wt._MAX_WRITE_CHARS + 1)})
        assert "太长" in r

    @pytest.mark.asyncio
    async def test_read_truncates(self, ws_root):
        (ws_root / "qq_111_private").mkdir(parents=True)
        (ws_root / "qq_111_private" / "long.txt").write_text("y" * 9000, encoding="utf-8")
        text = await wt.workspace_read.ainvoke({"path": "long.txt", "max_chars": 6000})
        assert "已截断" in text and len(text) < 9000


# ---------------------------------------------------------------- run_code 门禁

class _FakeResp:
    status_code = 200

    def json(self):
        return {"ok": True, "killed": False, "returncode": 0, "duration_ms": 12,
                "stdout": "hello", "stderr": "",
                "files": [{"path": "out.txt", "size": 5}]}


class _FakeClient:
    posts = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        type(self).posts.append((url, json))
        return _FakeResp()


@pytest.fixture()
def fake_sandbox(monkeypatch):
    _FakeClient.posts = []
    monkeypatch.setattr(wt.httpx, "AsyncClient", _FakeClient)
    return _FakeClient.posts


class TestRunCodeGate:
    """门禁误判回归：非管理员无批准一律拒；管理员/内核人审批准放行。"""

    @pytest.mark.asyncio
    async def test_non_admin_rejected(self, fake_sandbox, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_core import security
        t1 = security.current_user_id.set("111")          # 非管理员
        t2 = security.admin_privileged.set(False)
        try:
            r = await wt.run_code.ainvoke({"code": "print(1)"})
        finally:
            security.current_user_id.reset(t1)
            security.admin_privileged.reset(t2)
        assert "管理员" in r
        assert not fake_sandbox                              # 沙箱一次都没被调

    @pytest.mark.asyncio
    async def test_admin_allowed(self, fake_sandbox, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_core import security
        token = security.current_user_id.set("999")          # 管理员本人
        try:
            r = await wt.run_code.ainvoke({"code": "print('hello')"})
        finally:
            security.current_user_id.reset(token)
        assert "hello" in r and "out.txt" in r
        assert len(fake_sandbox) == 1
        url, payload = fake_sandbox[0]
        assert url.endswith("/run") and payload["timeout"] <= 30
        assert ":" not in payload["workdir"]                 # workdir 已清洗

    @pytest.mark.asyncio
    async def test_kernel_approved_step_allowed(self, fake_sandbox, monkeypatch):
        """非管理员但内核人审已批准的步骤 -> 放行（Phase 2 群友走审批路径）。"""
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_agent.task_kernel import executor
        from junjun_core import security
        t1 = security.current_user_id.set("111")
        t2 = executor._kernel_step_approved.set(True)
        try:
            r = await wt.run_code.ainvoke({"code": "print('ok')"})
        finally:
            security.current_user_id.reset(t1)
            executor._kernel_step_approved.reset(t2)
        assert "执行完成" in r and len(fake_sandbox) == 1

    @pytest.mark.asyncio
    async def test_static_scan_blocks_before_sandbox(self, fake_sandbox, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_core import security
        token = security.current_user_id.set("999")
        try:
            r = await wt.run_code.ainvoke({"code": "import os\nprint(1)"})
        finally:
            security.current_user_id.reset(token)
        assert "预检" in r and not fake_sandbox

    @pytest.mark.asyncio
    async def test_sandbox_down_raises_with_suggestion(self, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_core import security

        class _DownClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                raise wt.httpx.ConnectError("conn refused")

        monkeypatch.setattr(wt.httpx, "AsyncClient", _DownClient)
        token = security.current_user_id.set("999")
        try:
            r = await _call(wt.run_code, {"code": "print(1)"})
        finally:
            security.current_user_id.reset(token)
        assert "沙箱服务不可达" in r
        if r.startswith("[TOOL_ERROR"):        # 包装路径：换乘指引应进错误文本
            assert "沙箱服务没启动" in r


# ---------------------------------------------------------------- 内核审批门（executor 侧）

class TestKernelApprovalGate:
    def _plan(self, user_id, action="run_code"):
        from junjun_agent.task_kernel.plan import Step, TaskPlan
        return TaskPlan(goal="g", chat_id="qq:1:private", user_id=user_id,
                        steps=[Step(id="s1", action=action, desc="x")])

    def test_non_admin_run_code_forced_human(self, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_agent.task_kernel.executor import _apply_approval_gates
        plan = self._plan("123")
        _apply_approval_gates(plan)
        assert plan.steps[0].verify == "human"

    def test_admin_run_code_untouched(self, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_agent.task_kernel.executor import _apply_approval_gates
        plan = self._plan("999")
        _apply_approval_gates(plan)
        assert plan.steps[0].verify == "tool_ok"

    def test_unknown_user_gated(self, monkeypatch):
        """user_id 缺失（拿不准）一律按非管理员上闸——宁可误拦不可误放。"""
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_agent.task_kernel.executor import _apply_approval_gates
        plan = self._plan("")
        _apply_approval_gates(plan)
        assert plan.steps[0].verify == "human"

    def test_send_feed_gate_still_works(self, monkeypatch):
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_agent.task_kernel.executor import _apply_approval_gates
        plan = self._plan("999", action="send_feed")
        _apply_approval_gates(plan)
        assert plan.steps[0].verify == "human"               # 管理员也拦（发布类）

    def test_admin_other_tools_untouched(self, monkeypatch):
        """误判回归：普通工具（web_search）在任何身份下都不该被人审门误伤。"""
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_agent.task_kernel.executor import _apply_approval_gates
        for uid in ("999", "123", ""):
            plan = self._plan(uid, action="web_search")
            _apply_approval_gates(plan)
            assert plan.steps[0].verify == "tool_ok"

    @pytest.mark.asyncio
    async def test_run_step_sets_approved_context(self):
        """_run_step 包装：verify=human 且已批准的步骤执行期间放行位为 True。"""
        from junjun_agent.task_kernel import executor
        from junjun_agent.task_kernel.plan import Step, TaskPlan
        seen = {}

        async def _spy(self, plan, step):
            seen["flag"] = executor.kernel_step_approved()

        plan = TaskPlan(goal="g", chat_id="c", steps=[
            Step(id="s1", action="run_code", desc="x", verify="human", approved=True)])
        orig = executor.TaskKernel._run_step_inner
        executor.TaskKernel._run_step_inner = _spy
        try:
            await executor.kernel._run_step(plan, plan.steps[0])
        finally:
            executor.TaskKernel._run_step_inner = orig
        assert seen["flag"] is True
        assert executor.kernel_step_approved() is False      # 执行完已复位


# ---------------------------------------------------------------- workspace_send/delete

class TestWorkspaceSend:
    def _mkfile(self, ws_root, name="chart.png", size=100):
        d = ws_root / "qq_111_private"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"x" * size)

    @pytest.mark.asyncio
    async def test_image_goes_outbound(self, ws_root, monkeypatch):
        self._mkfile(ws_root, "chart.png")
        sent = []

        async def _fake_proactive(chat_id, segments, **kw):
            sent.append((chat_id, segments))
            return True

        import junjun_agent.outbound as ob
        monkeypatch.setattr(ob, "send_proactive", _fake_proactive)
        r = await wt.workspace_send.ainvoke({"path": "chart.png"})
        assert "已发到当前聊天" in r
        assert sent and sent[0][1][0].type == "image"
        assert sent[0][1][0].data.endswith("chart.png")

    @pytest.mark.asyncio
    async def test_docx_uploads_group_file(self, ws_root, monkeypatch):
        """非图片走文件上传；会话种类决定群文件还是私聊文件。"""
        self._mkfile(ws_root, "report.docx")
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:888:group")     # 切到群会话
        calls = []

        async def _fake_up(group_id, file_path, name=""):
            calls.append((group_id, name))
            return True

        monkeypatch.setattr(wt, "_ROOT", ws_root)        # 会话目录随 chat_id 变
        try:
            (ws_root / "qq_888_group").mkdir(parents=True)
            (ws_root / "qq_888_group" / "report.docx").write_bytes(b"doc")
            from junjun_core import napcat_client
            monkeypatch.setattr(napcat_client, "upload_group_file", _fake_up)
            r = await wt.workspace_send.ainvoke({"path": "report.docx"})
        finally:
            current_chat_id.reset(token)
        assert "群文件" in r
        assert calls == [("888", "report.docx")]

    @pytest.mark.asyncio
    async def test_missing_file(self, ws_root):
        r = await _call(wt.workspace_send, {"path": "ghost.png"})
        assert "没有" in r

    @pytest.mark.asyncio
    async def test_too_big(self, ws_root, monkeypatch):
        monkeypatch.setattr(wt, "_SEND_MAX_BYTES", 10)
        self._mkfile(ws_root, "big.png", size=100)
        r = await _call(wt.workspace_send, {"path": "big.png"})
        assert "太大" in r

    @pytest.mark.asyncio
    async def test_traversal_blocked(self, ws_root):
        r = await _call(wt.workspace_send, {"path": "../secret.png"})
        assert "拒绝" in r or "相对路径" in r


class TestWorkspaceDelete:
    @pytest.mark.asyncio
    async def test_delete_roundtrip(self, ws_root):
        d = ws_root / "qq_111_private"
        d.mkdir(parents=True)
        (d / "tmp.txt").write_text("x", encoding="utf-8")
        r = await wt.workspace_delete.ainvoke({"path": "tmp.txt"})
        assert "已删除" in r
        assert not (d / "tmp.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_missing_and_dir(self, ws_root):
        assert "没有" in await _call(wt.workspace_delete, {"path": "ghost.txt"})
        d = ws_root / "qq_111_private"
        (d / "subdir").mkdir(parents=True)
        assert "只能删文件" in await _call(wt.workspace_delete, {"path": "subdir"})


# ---------------------------------------------------------------- fetch_page SSRF

class TestFetchPageSsrf:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8100/health",
        "http://localhost/admin",
        "http://192.168.1.1/router",
        "ftp://example.com/x",
        "不是网址",
    ])
    def test_rejected(self, url):
        assert wt._ssrf_check(url) != ""

    @pytest.mark.asyncio
    async def test_tool_level_reject_no_network(self, ws_root):
        r = await wt.fetch_page.ainvoke({"url": "http://127.0.0.1/secret"})
        assert "抓不了" in r


# ---------------------------------------------------------------- 插件 anatomy

def test_manifest_and_tools():
    manifest = json.loads(
        (Path(wt.__file__).parent / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "workspace"
    assert manifest["tools_attr"] == "TOOLS"
    assert manifest["admin_only"] is False    # 门禁在 run_code 体内（框架门会误杀已批准步骤）
    names = [t.name for t in wt.TOOLS]
    assert names == ["run_code", "workspace_read", "workspace_write",
                     "workspace_list", "workspace_send", "workspace_delete",
                     "workspace_save_file", "fetch_page"]
    for t in wt.TOOLS:                        # docstring 铁律：≥15 字 + 何时使用
        doc = (t.description or "").strip()
        assert len(doc) >= 15 and "何时使用" in doc, t.name
