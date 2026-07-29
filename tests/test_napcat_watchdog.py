"""NapCat 看门狗检测逻辑测试（不触碰真实进程/网络）。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import napcat_watchdog as w  # noqa: E402


@pytest.fixture
def _napcat_dir(monkeypatch, tmp_path):
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(w, "NAPCAT_DIR", tmp_path)
    monkeypatch.setattr(w, "QQ", "1")
    return tmp_path


def _write_cfg(d, servers):
    (d / "config" / "onebot11_1.json").write_text(
        json.dumps({"network": {"httpServers": servers}}), encoding="utf-8")


class TestHttpEndpoint:
    def test_first_enabled_server(self, _napcat_dir):
        _write_cfg(_napcat_dir, [
            {"enable": False, "host": "127.0.0.1", "port": 1, "token": "x"},
            {"enable": True, "host": "127.0.0.1", "port": 3100, "token": "tok"},
        ])
        assert w._http_endpoint() == ("http://127.0.0.1:3100/get_status", "tok")

    def test_no_server_returns_none(self, _napcat_dir):
        _write_cfg(_napcat_dir, [])
        assert w._http_endpoint() is None

    def test_missing_file_returns_none(self, _napcat_dir):
        assert w._http_endpoint() is None


class TestCheckOnline:
    def _fake_urlopen(self, monkeypatch, payload=None, exc=None):
        class _Resp:
            def read(self): return json.dumps(payload).encode()
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def _urlopen(req, timeout=None):
            if exc:
                raise exc
            return _Resp()

        monkeypatch.setattr(w.urllib.request, "urlopen", _urlopen)

    def test_online(self, _napcat_dir, monkeypatch):
        _write_cfg(_napcat_dir, [{"enable": True, "host": "h", "port": 1, "token": ""}])
        self._fake_urlopen(monkeypatch, {"data": {"online": True, "good": True}})
        assert w.check_online() is True

    def test_kicked_offline(self, _napcat_dir, monkeypatch):
        _write_cfg(_napcat_dir, [{"enable": True, "host": "h", "port": 1, "token": ""}])
        self._fake_urlopen(monkeypatch, {"data": {"online": False, "good": False}})
        assert w.check_online() is False

    def test_unreachable(self, _napcat_dir, monkeypatch):
        _write_cfg(_napcat_dir, [{"enable": True, "host": "h", "port": 1, "token": ""}])
        self._fake_urlopen(monkeypatch, exc=ConnectionRefusedError())
        assert w.check_online() is False

    def test_no_server(self, _napcat_dir):
        _write_cfg(_napcat_dir, [])
        assert w.check_online() is False
