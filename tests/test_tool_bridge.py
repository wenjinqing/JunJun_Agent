"""沙箱工具桥测试（2026-08-15，PTC 试点；挂起-回放文件协议，--network=none 不动）。

核心断言：白名单宿主侧强制、身份/会话 context 按发起者设定、工具异常
不外溢、SDK 注入与协议文件正确、drive 回放循环端到端（含真子进程冒烟：
SDK 挂起写请求 -> 宿主执行 -> 缓存回放 -> 代码跑完）。
全部离线：不碰真沙箱服务、不写 data/。
"""

import json
import subprocess
import sys

import pytest

import junjun_skills.plugins.workspace.bridge as bridge
import junjun_skills.plugins.workspace.tools as wt


@pytest.fixture
def _bridge_on(monkeypatch):
    """桥启用 + 默认白名单。"""
    monkeypatch.setattr(bridge, "_cfg", lambda: {"tool_bridge": True})
    return bridge


class _StubTool:
    def __init__(self, name, fn):
        self.name = name
        self.description = f"{name} 桩"
        self._fn = fn

    async def ainvoke(self, args):
        return self._fn(args)


def _bind_tools(monkeypatch, tools):
    import junjun_skills.registry as reg
    monkeypatch.setattr(reg, "get_tools", lambda session=None: tools)


class TestExecute:
    @pytest.mark.asyncio
    async def test_non_whitelisted_rejected(self, _bridge_on, monkeypatch):
        """副作用工具即使注册了也不许过——白名单是唯一防线。"""
        called = []
        _bind_tools(monkeypatch, [
            _StubTool("set_reminder", lambda a: called.append(a) or "ok")])
        out = await bridge.execute("set_reminder", {}, chat_id="c", user_id="u")
        assert out["ok"] is False and "白名单" in out["error"]
        assert not called

    @pytest.mark.asyncio
    async def test_whitelisted_call_with_context(self, _bridge_on, monkeypatch):
        seen = {}

        def _fn(args):
            from junjun_skills.builtin.memory_skills import current_chat_id
            from junjun_core.security import current_user_id
            seen["chat"] = current_chat_id.get("")
            seen["user"] = current_user_id.get("")
            seen["args"] = args
            return "搜索结果文本"

        _bind_tools(monkeypatch, [_StubTool("web_search", _fn)])
        out = await bridge.execute("web_search", {"query": "固态电池"},
                                   chat_id="qq:42:group", user_id="10000")
        assert out == {"ok": True, "text": "搜索结果文本"}
        assert seen == {"chat": "qq:42:group", "user": "10000",
                        "args": {"query": "固态电池"}}

    @pytest.mark.asyncio
    async def test_tool_exception_contained(self, _bridge_on, monkeypatch):
        def _boom(args):
            raise RuntimeError("炸了")
        _bind_tools(monkeypatch, [_StubTool("get_time", _boom)])
        out = await bridge.execute("get_time", {}, chat_id="c", user_id="u")
        assert out["ok"] is False and "炸了" in out["error"]

    @pytest.mark.asyncio
    async def test_text_capped(self, _bridge_on, monkeypatch):
        _bind_tools(monkeypatch, [_StubTool("get_time", lambda a: "长" * 40000)])
        out = await bridge.execute("get_time", {}, chat_id="c", user_id="u")
        assert out["ok"] and len(out["text"]) == 16000


class TestSdkInstall:
    def test_sdk_file_written(self, _bridge_on, tmp_path):
        assert bridge.install_sdk(tmp_path)
        content = (tmp_path / "jjtools.py").read_text(encoding="utf-8")
        assert "def web_search" in content and "def call" in content
        assert "sys.exit(42)" in content          # 挂起约定
        assert ".jj_request.json" in content

    def test_sdk_replay_protocol_subprocess(self, _bridge_on, tmp_path):
        """真子进程冒烟：挂起写请求 -> 缓存就位后重放 -> 拿到结果正常退出。
        不起 Docker 也验证协议两端（SDK 文件本身 + 缓存语义）。"""
        bridge.install_sdk(tmp_path)
        script = "import jjtools; print('TIME=' + jjtools.get_time())"

        first = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=30)
        assert first.returncode == 42                       # 挂起
        req = json.loads((tmp_path / ".jj_request.json").read_text(encoding="utf-8"))
        assert req == {"idx": 0, "tool": "get_time", "args": {}}

        # 宿主执行完毕 -> 写缓存 -> 重放
        (tmp_path / ".jj_request.json").unlink()
        (tmp_path / ".jj_cache.json").write_text(
            json.dumps([{"ok": True, "text": "2026-08-15 21:00"}]),
            encoding="utf-8")
        second = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                                capture_output=True, text=True, timeout=30)
        assert second.returncode == 0
        assert "TIME=2026-08-15 21:00" in second.stdout

    def test_sdk_cached_error_reraised(self, _bridge_on, tmp_path):
        """缓存里的失败结果在重放时原样抛回给代码。"""
        bridge.install_sdk(tmp_path)
        (tmp_path / ".jj_cache.json").write_text(
            json.dumps([{"ok": False, "error": "工具挂了"}]), encoding="utf-8")
        script = ("import jjtools\n"
                  "try:\n    jjtools.get_time()\nexcept RuntimeError as e:\n"
                  "    print('CAUGHT', e)")
        r = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and "CAUGHT 工具桥调用失败: 工具挂了" in r.stdout


class TestDrive:
    @pytest.mark.asyncio
    async def test_suspend_execute_replay(self, _bridge_on, monkeypatch,
                                          tmp_path):
        """挂起-回放主循环：第一轮 exit42+请求文件 -> 宿主执行 -> 第二轮跑完。"""
        executed = []

        def _fn(args):
            executed.append(args)
            return "时间到"

        _bind_tools(monkeypatch, [_StubTool("get_time", _fn)])
        rounds = []

        async def _post():
            rounds.append(1)
            if len(rounds) == 1:
                (tmp_path / ".jj_request.json").write_text(json.dumps(
                    {"idx": 0, "tool": "get_time", "args": {}}), encoding="utf-8")
                return {"returncode": 42, "stdout": "", "stderr": ""}
            return {"returncode": 0, "stdout": "TIME=时间到", "stderr": ""}

        resp = await bridge.drive(_post, tmp_path, chat_id="qq:1:group",
                                  user_id="10000")
        assert resp["returncode"] == 0 and "TIME=时间到" in resp["stdout"]
        assert executed == [{}]
        cache = json.loads((tmp_path / ".jj_cache.json").read_text(encoding="utf-8"))
        assert cache == [{"ok": True, "text": "时间到"}]
        assert not (tmp_path / ".jj_request.json").exists()   # 请求已消费

    @pytest.mark.asyncio
    async def test_max_calls_cap(self, _bridge_on, monkeypatch, tmp_path):
        """超限后注入错误结果让代码自己收场；仍挂起则带说明返回。"""
        monkeypatch.setattr(bridge, "_max_calls", lambda: 1)
        executed = []
        _bind_tools(monkeypatch, [
            _StubTool("get_time", lambda a: executed.append(a) or "t")])

        async def _post():
            # 代码每次都发起新调用（模拟不知收手的编排）
            (tmp_path / ".jj_request.json").write_text(json.dumps(
                {"idx": 0, "tool": "get_time", "args": {}}), encoding="utf-8")
            return {"returncode": 42, "stdout": "", "stderr": ""}

        resp = await bridge.drive(_post, tmp_path, chat_id="c", user_id="u")
        assert len(executed) == 1                              # 只真调一次
        assert "上限" in resp["stderr"]

    @pytest.mark.asyncio
    async def test_stale_request_cleared(self, _bridge_on, monkeypatch,
                                         tmp_path):
        """开场清残留：上次崩掉的请求/缓存不污染新一轮。"""
        (tmp_path / ".jj_request.json").write_text("{}")
        (tmp_path / ".jj_cache.json").write_text("[]")

        async def _post():
            return {"returncode": 0, "stdout": "ok", "stderr": ""}

        resp = await bridge.drive(_post, tmp_path, chat_id="c", user_id="u")
        assert resp["returncode"] == 0
        assert not (tmp_path / ".jj_request.json").exists()
        assert not (tmp_path / ".jj_cache.json").exists()


class TestRunCodeInjection:
    @pytest.mark.asyncio
    async def test_run_code_injects_sdk_and_drives(
            self, _bridge_on, monkeypatch, tmp_path):
        """桥开启 + 管理员调 run_code → SDK 注入 + 挂起回放一轮跑完。"""
        monkeypatch.setattr(wt, "_ROOT", tmp_path / "ws")
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_core import security
        from junjun_skills.builtin.memory_skills import current_chat_id

        _bind_tools(monkeypatch, [_StubTool("get_time", lambda a: "21:00")])
        posts = []

        class _Resp:
            status_code = 200

            def __init__(self, data):
                self._data = data

            def json(self):
                return self._data

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                posts.append(json)
                wd = tmp_path / "ws" / "qq_777_group"
                if len(posts) == 1:   # 第一轮：沙箱挂起
                    (wd / ".jj_request.json").write_text(
                        __import__("json").dumps(
                            {"idx": 0, "tool": "get_time", "args": {}}),
                        encoding="utf-8")
                    return _Resp({"killed": False, "returncode": 42,
                                  "duration_ms": 5, "stdout": "",
                                  "stderr": "", "files": []})
                return _Resp({"killed": False, "returncode": 0,
                              "duration_ms": 5, "stdout": "TIME=21:00",
                              "stderr": "",
                              "files": [{"path": "jjtools.py", "size": 100},
                                        {"path": "out.txt", "size": 3}]})

        monkeypatch.setattr(wt, "make_async_client", lambda **kw: _Client())
        t1 = security.current_user_id.set("999")
        t2 = current_chat_id.set("qq:777:group")
        try:
            r = await wt.run_code.ainvoke({"code": "import jjtools; print('TIME='+jjtools.get_time())"})
        finally:
            security.current_user_id.reset(t1)
            current_chat_id.reset(t2)
        assert len(posts) == 2                               # 挂起一次+回放一次
        assert "TIME=21:00" in r
        assert "jjtools.py" not in r                         # 协议文件不出现在文件清单
        assert "out.txt" in r
        sdk = tmp_path / "ws" / "qq_777_group" / "jjtools.py"
        assert sdk.is_file()

    @pytest.mark.asyncio
    async def test_run_code_no_sdk_when_bridge_off(self, monkeypatch, tmp_path):
        """桥默认关：不注入 SDK 不走回放（现有行为不变）。"""
        monkeypatch.setattr(wt, "_ROOT", tmp_path / "ws")
        monkeypatch.setattr(bridge, "_cfg", lambda: {})
        monkeypatch.setenv("ADMIN_QQ", "999")
        from junjun_core import security
        from junjun_skills.builtin.memory_skills import current_chat_id

        posts = []

        class _Resp:
            status_code = 200

            def json(self):
                return {"killed": False, "returncode": 0, "duration_ms": 5,
                        "stdout": "ok", "stderr": "", "files": []}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                posts.append(json)
                return _Resp()

        monkeypatch.setattr(wt, "make_async_client", lambda **kw: _Client())
        t1 = security.current_user_id.set("999")
        t2 = current_chat_id.set("qq:777:group")
        try:
            await wt.run_code.ainvoke({"code": "print(1)"})
        finally:
            security.current_user_id.reset(t1)
            current_chat_id.reset(t2)
        assert len(posts) == 1                               # 单轮直跑
        assert not (tmp_path / "ws" / "qq_777_group" / "jjtools.py").exists()
