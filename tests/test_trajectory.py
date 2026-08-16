"""轨迹日志测试（2026-08-15，DSH 追记式会话日志移植）。

核心断言：emit 落 JSONL 行、字段齐、开关生效、目录故障绝不炸主流程、
长字段兜底截断。测试全部指向临时目录——绝不写 data/。
"""

import json

import pytest

import junjun_core.observability.trajectory as trajectory


@pytest.fixture
def _tmp_traj(monkeypatch, tmp_path):
    monkeypatch.setattr(trajectory, "_dir", lambda: tmp_path)
    monkeypatch.setattr(trajectory, "_enabled", lambda: True)
    return tmp_path


def _read_lines(d):
    files = list(d.glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines()]


class TestEmit:
    def test_writes_jsonl_line(self, _tmp_traj):
        trajectory.emit("inbound", "qq:1:group", user="某人", text="你好")
        (rec,) = _read_lines(_tmp_traj)
        assert rec["kind"] == "inbound"
        assert rec["chat_id"] == "qq:1:group"
        assert rec["user"] == "某人"
        assert rec["ts"] > 0

    def test_appends_same_day_file(self, _tmp_traj):
        trajectory.emit("inbound", "c1")
        trajectory.emit("tk_step", "c1", step="s1", status="done")
        recs = _read_lines(_tmp_traj)
        assert [r["kind"] for r in recs] == ["inbound", "tk_step"]

    def test_disabled_writes_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(trajectory, "_dir", lambda: tmp_path)
        monkeypatch.setattr(trajectory, "_enabled", lambda: False)
        trajectory.emit("inbound", "c1", text="x")
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_never_raises_on_bad_dir(self, monkeypatch):
        def _boom():
            raise OSError("只读文件系统")
        monkeypatch.setattr(trajectory, "_dir", _boom)
        monkeypatch.setattr(trajectory, "_enabled", lambda: True)
        trajectory.emit("inbound", "c1")          # 不许抛

    def test_long_field_truncated(self, _tmp_traj):
        trajectory.emit("inbound", "c1", text="长" * 800)
        (rec,) = _read_lines(_tmp_traj)
        assert len(rec["text"]) < 600 and rec["text"].endswith("…")
