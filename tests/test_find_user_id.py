"""find_user_id 昵称解析 + relationship MCP 昵称容忍测试。"""

import pytest

from junjun_skills.builtin.capability_skills import find_user_id
from junjun_skills.builtin.memory_skills import current_chat_id


class _Pred:
    """可组合的行过滤器（模拟 peewee 表达式的 & 组合）。"""
    def __init__(self, fn): self.fn = fn
    def __call__(self, r): return self.fn(r)
    def __and__(self, other): return _Pred(lambda r: self.fn(r) and other.fn(r))


class _Field:
    def __init__(self, attr): self.attr = attr
    def contains(self, s): return _Pred(lambda r: s in getattr(r, self.attr))
    def __ne__(self, o): return _Pred(lambda r: getattr(r, self.attr) != o)
    def __eq__(self, o): return _Pred(lambda r: getattr(r, self.attr) == o)
    def desc(self): return self


class _Row:
    def __init__(self, uid, nick, chat_id, ts=0):
        self.user_id = uid
        self.user_nickname = nick
        self.chat_id = chat_id
        self.is_bot = False
        self.time = ts


def _fake_messages(rows):
    class _Q:
        def __init__(self, rs): self._rs = list(rs)
        def where(self, *preds):
            for p in preds:
                if callable(p):
                    self._rs = [r for r in self._rs if p(r)]
            return self
        def order_by(self, *a): return self
        def limit(self, n): return self._rs[:n]
        def first(self): return self._rs[0] if self._rs else None

    class _M:
        user_nickname = _Field("user_nickname")
        is_bot = _Field("is_bot")
        user_id = _Field("user_id")
        time = _Field("time")

        @staticmethod
        def select(): return _Q(rows)

    return _M


@pytest.fixture(autouse=True)
def _chat_ctx():
    token = current_chat_id.set("qq:1054390069:group")
    yield
    current_chat_id.reset(token)


class TestFindUserId:
    def test_exact_match(self, monkeypatch):
        import junjun_core.database as db
        rows = [_Row("2991064865", "鹤", "qq:1054390069:group")]
        monkeypatch.setattr(db, "Messages", _fake_messages(rows))
        out = find_user_id.invoke({"nickname": "鹤"})
        assert "2991064865" in out and "当前会话" in out

    def test_digit_passthrough(self):
        out = find_user_id.invoke({"nickname": "2991064865"})
        assert "本身就是 QQ 号" in out

    def test_not_found(self, monkeypatch):
        import junjun_core.database as db
        monkeypatch.setattr(db, "Messages", _fake_messages([]))
        out = find_user_id.invoke({"nickname": "不存在的人"})
        assert "没找到" in out

    def test_multiple_candidates(self, monkeypatch):
        import junjun_core.database as db
        rows = [_Row("111", "小白兔", "qq:1054390069:group"),
                _Row("222", "大白兔", "qq:1054390069:group"),
                _Row("333", "兔斯基", "qq:1054390069:group")]
        monkeypatch.setattr(db, "Messages", _fake_messages(rows))
        out = find_user_id.invoke({"nickname": "兔"})
        assert "111" in out and "222" in out
        assert "确认是哪一个" in out


class TestMcpResolve:
    """relationship MCP server 的昵称容忍（直接调模块函数，不起 stdio）。"""

    @pytest.fixture
    def _server(self, monkeypatch):
        import importlib
        import junjun_core.database as db
        rows = [_Row("2991064865", "鹤", "qq:1054390069:group")]
        monkeypatch.setattr(db, "Messages", _fake_messages(rows))
        import junjun_mcp_server.relationship_mcp_server as srv
        return importlib.import_module("junjun_mcp_server.relationship_mcp_server")

    def test_resolve_digit_passthrough(self, _server):
        assert _server._resolve_user_id("qq", "2991064865") == "2991064865"

    def test_resolve_nickname(self, _server):
        assert _server._resolve_user_id("qq", "鹤") == "2991064865"

    def test_resolve_garbage_returns_original(self, _server):
        assert _server._resolve_user_id("qq", "鹤的QQ号") == "鹤的QQ号"

    def test_penalty_rejects_unresolved(self, _server):
        out = _server.apply_relationship_penalty(
            user_id="鹤的QQ号", platform="qq", penalty_type="harassment")
        assert "find_user_id" in out

    def test_penalty_accepts_nickname(self, _server, monkeypatch):
        added = {}

        class _Store:
            def add_point(self, platform, uid, cat, content, weight=0.0):
                added["uid"] = uid
                added["content"] = content

        monkeypatch.setattr(_server, "_store", lambda: _Store())
        out = _server.apply_relationship_penalty(
            user_id="鹤", platform="qq", penalty_type="harassment",
            severity="minor", reason="频繁摸头调戏")
        assert added["uid"] == "2991064865"
        assert "2991064865" in out
