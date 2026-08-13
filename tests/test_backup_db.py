"""backup_db 0-token 测试：WAL 未 checkpoint 数据不丢、轮换只留 7 代、完整性校验。

全程 tmp 目录造库，绝不碰 data/junjun.db（CLAUDE.md 硬约束）。
"""

import sqlite3
import zipfile

import pytest

from scripts import backup_db as b


def _make_db(path, rows=10, wal=True):
    conn = sqlite3.connect(str(path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"行{i}") for i in range(rows)])
    conn.commit()   # WAL 模式下 commit 后数据在 wal 文件里，未必 checkpoint 进主库
    return conn     # 故意不关——保持 WAL 未回收状态，模拟生产库热备份


class TestBackup:
    def test_wal_data_not_lost(self, tmp_path):
        """P0 核心：WAL 里未落主库的数据必须进备份（复制文件做不到，.backup 可以）。"""
        src = tmp_path / "junjun.db"
        conn = _make_db(src, rows=42)
        dest = b.backup(src, tmp_path / "bk")
        out = sqlite3.connect(f"file:{dest.as_posix()}?mode=ro", uri=True)
        assert out.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 42
        assert out.execute("SELECT v FROM t WHERE id=41").fetchone()[0] == "行41"
        out.close()
        conn.close()

    def test_rotation_keeps_seven(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        src = tmp_path / "junjun.db"
        conn = _make_db(src)
        # 只换 backup_db 模块看到的 time 命名空间——patch 全局 time.strftime
        # 会把 logging 内部的 strftime(fmt, t) 两参调用搞炸
        for i in range(9):
            monkeypatch.setattr(b, "time", SimpleNamespace(
                strftime=lambda fmt, _i=i: f"202608{_i+10:02d}_120000"))
            b.backup(src, tmp_path / "bk")
        assert len(list((tmp_path / "bk").glob("junjun_*.db"))) == 7
        names = sorted(p.name for p in (tmp_path / "bk").glob("junjun_*.db"))
        assert names[0] == "junjun_20260812_120000.db"   # 最老的两代被轮换删掉
        conn.close()

    def test_memory_packed(self, tmp_path, monkeypatch):
        src = tmp_path / "junjun.db"
        conn = _make_db(src)
        mem = tmp_path / "mem"
        mem.mkdir()
        (mem / "long_term.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(b, "MEMORY_DIR", mem)
        b.backup(src, tmp_path / "bk")
        zips = list((tmp_path / "bk").glob("memory_*.zip"))
        assert len(zips) == 1
        assert "long_term.json" in zipfile.ZipFile(zips[0]).namelist()
        conn.close()

    def test_missing_src_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            b.backup(tmp_path / "没有.db", tmp_path / "bk")

    def test_same_drive_warns(self, tmp_path, capsys):
        """同盘备份必须告警（盘坏=两者俱失）——这条 warn 是 P0 审查的核心诉求之一。
        仓库 logger 走 structlog PrintLogger（stdout），不是 stdlib logging——capsys 接。"""
        src = tmp_path / "junjun.db"
        conn = _make_db(src)
        b.backup(src, tmp_path / "bk")
        conn.close()
        assert "同一磁盘" in capsys.readouterr().out
