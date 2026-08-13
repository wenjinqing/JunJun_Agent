"""sandbox/server.py 纯逻辑 0-token 测试：workdir 归属断言 + token 鉴权 + 清洗契约。

2026-08-13 审查 P0 回归：_safe 放行点号，workdir=".." 可把整个 data/ 挂进容器。
不起 docker——测的是编排层的路径与鉴权逻辑（边界本身由容器冒烟覆盖）。
"""

import pytest
from fastapi.testclient import TestClient

from sandbox import server


class TestWorkdirGuard:
    # 清洗把 / \ 换成 _，混分隔符的穿越串会变成无害目录名——真正的穿越向量
    # 只有纯点号 ".." / "."（P0 实锤路径）
    @pytest.mark.parametrize("w", ["..", "."])
    def test_traversal_rejected(self, w):
        with pytest.raises(ValueError):
            server._resolve_workdir(w)

    @pytest.mark.parametrize("w", ["default", "qq_999_group", "a.b.c", "任务-x",
                                   "....//....", "..\\..\\data", "../.."])
    def test_normal_accepted(self, w):
        wd = server._resolve_workdir(w)
        assert server.ROOT in wd.parents
        assert wd != server.ROOT

    def test_dotdot_exactly_was_the_p0(self):
        """P0 实锤路径单独钉死：'..' 经 _safe 原样穿过，必须被归属断言拦下。"""
        assert server._safe("..") == ".."     # 清洗确实放行点号（契约如实记录）
        with pytest.raises(ValueError):
            server._resolve_workdir("..")


class TestSafeNameContract:
    """跨进程契约锁：插件侧 _safe_name 与沙箱侧 _safe 必须逐字一致——
    单侧漂移 = 插件写 data/workspace/<safe(chat_id)> 而沙箱解析到别处。"""

    @pytest.mark.parametrize("s", ["qq:999:group", "qq:111:private", "", "a/b\\c:d",
                                   "很长的" * 30, ".", "..", "群 名 带 空 格"])
    def test_same_rule(self, s):
        from junjun_skills.plugins.workspace.tools import _safe_name
        assert server._safe(s) == _safe_name(s)


class TestTokenAuth:
    def test_token_enforced_when_set(self, monkeypatch):
        monkeypatch.setattr(server, "_TOKEN", "sekrit")
        client = TestClient(server.app)
        r = client.post("/run", json={"code": "print(1)"})
        assert r.status_code == 401
        r = client.post("/run", json={"code": "print(1)"},
                        headers={"X-Sandbox-Token": "wrong"})
        assert r.status_code == 401

    def test_bad_workdir_400_before_docker(self, monkeypatch):
        """workdir 非法在起容器前就 400——docker 不可用的测试环境也能验证。"""
        monkeypatch.setattr(server, "_TOKEN", "")
        client = TestClient(server.app)
        r = client.post("/run", json={"code": "print(1)", "workdir": ".."})
        assert r.status_code == 400
        assert "workdir" in r.text
