"""jrys 插件测试：jrrp 确定性 / 每日签复用 / 卡片渲染 / LLM 降级 / 分项与详版。"""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text):
    return SimpleNamespace(text=text, user_id="12345", nickname="甲", at_bot=False, message_id="m1")


def _ctx(text, is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


@pytest.fixture
def _jrys(monkeypatch, tmp_path):
    """导入插件并把数据目录隔离到 tmp_path（reload 以在清空总线后重新注册命令）。"""
    import importlib

    import junjun_skills.plugins.jrys.tools as jrys
    jrys = importlib.reload(jrys)
    monkeypatch.setattr(jrys, "DATA_DIR", tmp_path)
    return jrys


def _fake_llm_text(prompt):
    """按提示词类型返回固定签文/分项星级（异步版本见各测试内 monkeypatch）。"""
    if "分项星级" in prompt:
        return "综合: 4\n桃花: 5\n工作: 3\n财运: 2\n学业: 1"
    if "STAR" in prompt:
        return "STAR: 4\nLINE: 甲，今日这一项小有起色，稳住节奏就好。"
    return "VERSE: 云开见月，事缓则圆，心定功成\nLINE: 甲，今天节奏放慢一点，事情反而会顺。"


class TestJrrp:
    def test_deterministic_same_args(self, _jrys):
        a = _jrys.jrrp("12345", "2026-07-21")
        b = _jrys.jrrp("12345", "2026-07-21")
        assert a == b and 0 <= a <= 100

    def test_varies_by_date_and_user(self, _jrys):
        # 多个日期/用户取样，结果不应全部相同（确定性但随输入变化）
        by_date = {_jrys.jrrp("12345", f"2026-07-{d:02d}") for d in range(1, 8)}
        by_user = {_jrys.jrrp(f"u{i}", "2026-07-21") for i in range(8)}
        assert len(by_date) > 1
        assert len(by_user) > 1
        assert all(0 <= v <= 100 for v in by_date | by_user)


class TestToday:
    async def test_full_chain_sends_text_and_image(self, _jrys, _fake_gateway, monkeypatch):
        async def _llm(prompt):
            return _fake_llm_text(prompt)

        monkeypatch.setattr(_jrys, "_ask_llm", _llm)
        result = await _jrys.jrys_today_cmd(_ctx("今日运势"))
        assert result is None
        assert len(_fake_gateway) == 1
        segs = _fake_gateway[0].segments
        assert segs[0].type == "text" and "【" in segs[0].data and "甲" in segs[0].data
        assert segs[1].type == "image"
        img = _jrys.Path(segs[1].data)
        assert img.exists() and img.suffix == ".png"
        assert img.read_bytes()[:4] == b"\x89PNG"

    async def test_same_day_reuse_no_repeat_llm(self, _jrys, _fake_gateway, monkeypatch):
        calls = []

        async def _llm(prompt):
            calls.append(prompt)
            return _fake_llm_text(prompt)

        monkeypatch.setattr(_jrys, "_ask_llm", _llm)
        await _jrys.jrys_today_cmd(_ctx("今日运势"))
        await _jrys.jrys_today_cmd(_ctx("今日运势"))
        assert len(calls) == 1  # 第二次复用记录，不再调 LLM
        # 两次回复签档一致
        t1 = _fake_gateway[0].segments[0].data
        t2 = _fake_gateway[1].segments[0].data
        assert t1.split("】")[0] == t2.split("】")[0]

    async def test_llm_failure_falls_back_to_local_quote(self, _jrys, _fake_gateway, monkeypatch):
        async def _none(prompt):
            return None

        monkeypatch.setattr(_jrys, "_ask_llm", _none)
        await _jrys.jrys_today_cmd(_ctx("今日运势"))
        segs = _fake_gateway[0].segments
        assert segs[1].type == "image"
        # 落盘记录的签文来自本地签文库
        rec_file = _jrys.DATA_DIR / f"12345_{date.today().isoformat()}.json"
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        local_lines = {e["line"] for e in _jrys.FORTUNE_QUOTES}
        assert rec["line"] in local_lines
        assert 0 <= rec["jrrp"] <= 100

    async def test_dispatch_keyword_routing(self, _jrys, _fake_gateway, monkeypatch):
        async def _llm(prompt):
            return _fake_llm_text(prompt)

        monkeypatch.setattr(_jrys, "_ask_llm", _llm)
        handled = await commands.dispatch(_session(), _meta("今日运势"))
        assert handled is True
        assert _fake_gateway[0].segments[1].type == "image"


class TestDetail:
    async def test_detail_stars_in_range(self, _jrys, _fake_gateway, monkeypatch):
        async def _llm(prompt):
            return _fake_llm_text(prompt)

        monkeypatch.setattr(_jrys, "_ask_llm", _llm)
        await _jrys.jrys_detail_cmd(_ctx("今日运势详"))
        assert _fake_gateway[0].segments[1].type == "image"
        rec_file = _jrys.DATA_DIR / f"12345_{date.today().isoformat()}.json"
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        assert rec["sub_stars"] == {"综合": 4, "桃花": 5, "工作": 3, "财运": 2, "学业": 1}
        assert all(1 <= v <= 5 for v in rec["sub_stars"].values())

    async def test_detail_fallback_deterministic_stars(self, _jrys, _fake_gateway, monkeypatch):
        async def _none(prompt):
            return None

        monkeypatch.setattr(_jrys, "_ask_llm", _none)
        await _jrys.jrys_detail_cmd(_ctx("今日运势详细"))
        rec_file = _jrys.DATA_DIR / f"12345_{date.today().isoformat()}.json"
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        assert set(rec["sub_stars"]) == {"综合", "桃花", "工作", "财运", "学业"}
        assert all(1 <= v <= 5 for v in rec["sub_stars"].values())
        # 降级星级确定性：直接重摇结果一致
        again = await _jrys._roll_sub_stars(rec["title"], rec["stars"], "12345", rec["date"])
        assert again == rec["sub_stars"]


class TestSingleDim:
    @pytest.mark.parametrize("keyword,dim", [("今日桃花", "桃花"), ("今日工作", "工作"),
                                             ("今日财运", "财运"), ("今日学业", "学业")])
    async def test_single_dim_sends_image(self, _jrys, _fake_gateway, monkeypatch,
                                          keyword, dim):
        async def _llm(prompt):
            return _fake_llm_text(prompt)

        monkeypatch.setattr(_jrys, "_ask_llm", _llm)
        handled = await commands.dispatch(_session(), _meta(keyword))
        assert handled is True
        segs = _fake_gateway[-1].segments
        assert segs[1].type == "image"
        assert f"今日{dim}" not in segs[0].data or "【" in segs[0].data  # 文本为摘要
        assert _jrys.Path(segs[1].data).exists()

    async def test_single_dim_llm_failure_fallback(self, _jrys, _fake_gateway, monkeypatch):
        async def _none(prompt):
            return None

        monkeypatch.setattr(_jrys, "_ask_llm", _none)
        await _jrys.jrys_taohua_cmd(_ctx("今日桃花"))
        segs = _fake_gateway[0].segments
        assert segs[1].type == "image"
        assert "本地兜底" in segs[0].data
