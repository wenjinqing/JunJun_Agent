"""日志落盘 0-token 测试（2026-08-13 审查 P1）：tee 双写 / ANSI 剥离 / 轮转 /
写盘异常不拖死业务 / capsys 兼容（动态 sys.stdout）。
"""

import pytest

from junjun_core.observability import logger as lg


@pytest.fixture()
def tee(tmp_path):
    t = lg._TeeStream(tmp_path / "bot.log")
    yield t, tmp_path / "bot.log"
    try:
        t._f.close()
    except Exception:
        pass


class TestTeeStream:
    def test_writes_to_file_and_stdout(self, tee, capsys):
        t, path = tee
        t.write("你好一行\n")
        t.flush()
        assert "你好一行" in path.read_text(encoding="utf-8")
        assert "你好一行" in capsys.readouterr().out   # 动态 stdout，capsys 可见

    def test_ansi_stripped_in_file_only(self, tee, capsys):
        t, path = tee
        t.write("\x1b[32m绿色\x1b[0m\n")
        assert "\x1b" not in path.read_text(encoding="utf-8")
        assert "绿色" in path.read_text(encoding="utf-8")
        assert "\x1b[32m" in capsys.readouterr().out   # 控制台保留颜色

    def test_rotation(self, tee, monkeypatch):
        t, path = tee
        monkeypatch.setattr(lg, "_MAX_BYTES", 100)
        for i in range(8):
            t.write(f"第{i}行" + "x" * 40 + "\n")
        assert path.with_name("bot.log.1").exists()   # 至少轮转过一次
        assert path.exists() and path.stat().st_size < 200

    def test_rotation_backups_chain(self, tee, monkeypatch):
        t, path = tee
        monkeypatch.setattr(lg, "_MAX_BYTES", 50)
        for i in range(40):
            t.write(f"line{i}" + "y" * 30 + "\n")
        # 备份链不超过 _BACKUPS 代
        backups = sorted(path.parent.glob("bot.log.*"))
        assert len(backups) <= lg._BACKUPS

    def test_broken_file_never_raises(self, tee):
        t, _ = tee
        t._f.close()              # 模拟盘写故障
        t.write("不该炸\n")       # 静默降级
        t.flush()


class TestInitialize:
    def test_log_name_none_skips_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lg, "_initialized", False)
        monkeypatch.setattr(lg, "_LOG_DIR", tmp_path)
        lg.initialize_logging("INFO", log_name=None)
        assert not list(tmp_path.glob("*.log"))
        monkeypatch.setattr(lg, "_initialized", False)
        lg.initialize_logging("INFO", log_name="bot")   # 恢复供后续测试
