"""catcafe 站点管理插件测试（无网络：_request 全部打桩；不触 DB）。

覆盖：读内容/读统计的文案格式、发公告的 read-modify-write 全链路
（插入最前+旧公告保留）、tag 白名单、1MB 上限、saved 确认、管理员门。
"""

import json

import pytest

import junjun_skills.plugins.catcafe.tools as tools
from junjun_core import security


@pytest.fixture
def site(monkeypatch):
    """打桩 _request：记录调用；GET 回站点样例（深拷贝），PUT 回 saved。"""
    calls = []
    data = {
        "title": "小涩猫咖啡厅", "author": "爱丽丝猫猫酱",
        "slogan": "咖啡厅长期营业中~", "authorStatus": "赶稿中",
        "notices": [{"date": "2026-08-06", "tag": "公告", "text": "旧公告"}],
        "novels": [{"title": "某小说", "desc": "d", "cup": "中杯 · 微糖",
                    "cat": "已完结", "file": "x.txt"}],
        "gallery": [{"img": "a.png", "title": "图", "note": ""}],
        "fanClub": {"name": "后援会", "desc": "d", "qq": "1"},
    }
    msgs = [
        {"nick": "路人甲", "content": "店长什么时候更新呀",
         "time": "2026-08-18 10:00"},
        {"nick": "小猪", "content": "打卡~\n今天也来撸猫",
         "time": "2026-08-17 22:00",
         "reply": "欢迎常来~", "replyBy": "猪咪君君"},
    ]

    async def fake(method, path, raw_body=None):
        calls.append({"method": method, "path": path,
                      "payload": json.loads(raw_body) if raw_body else None})
        if path.endswith("/stats"):
            return {"visits": 172, "pets": 908,
                    "day": {"date": "2026-08-18", "visits": 5},
                    "topPost": {"nick": "系七月狐吖", "likes": 12}}, None
        if path == "/api/messages":
            return json.loads(json.dumps(msgs)), None
        if path.endswith("/messages/reply") or path.endswith("/messages/delete"):
            return {"status": "ok"}, None
        if method == "GET":
            return json.loads(json.dumps(data)), None
        return {"status": "saved"}, None

    monkeypatch.setattr(tools, "_request", fake)
    return {"calls": calls, "data": data, "msgs": msgs}


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(security, "is_admin_privileged", lambda: True)


@pytest.fixture
def non_admin(monkeypatch):
    monkeypatch.setattr(security, "is_admin_privileged", lambda: False)


class TestReadTools:
    async def test_get_content_summary(self, site):
        out = await tools.catcafe_get_content.ainvoke({})
        assert "小涩猫咖啡厅" in out and "旧公告" in out
        assert "某小说" in out and "公告栏 1 条" in out

    async def test_get_stats(self, site):
        out = await tools.catcafe_get_stats.ainvoke({})
        assert "访问：172" in out and "撸猫：908" in out
        assert "今日（2026-08-18）" in out and "系七月狐吖" in out

    async def test_error_passthrough(self, monkeypatch):
        async def down(method, path, raw_body=None):
            return None, "连不上站点接口（ConnectError），稍后再试。"
        monkeypatch.setattr(tools, "_request", down)
        out = await tools.catcafe_get_content.ainvoke({})
        assert "连不上站点接口" in out

    async def test_missing_key_short_circuit(self, monkeypatch):
        """key 未配置：不发 HTTP，直接给可读错误（真实 _request，无网络）。"""
        monkeypatch.delenv("CATCAFE_API_KEY", raising=False)
        result, err = await tools._request("GET", "/api/agent/content")
        assert result is None and "未配置" in err


class TestPostNotice:
    async def test_prepends_and_keeps_old(self, site, admin):
        out = await tools.catcafe_post_notice.ainvoke(
            {"text": "新坑开张啦", "tag": "新坑"})
        assert "完成" in out
        assert [c["method"] for c in site["calls"]] == ["GET", "PUT"]
        payload = site["calls"][1]["payload"]
        assert payload["notices"][0]["tag"] == "新坑"
        assert payload["notices"][0]["text"] == "新坑开张啦"
        assert payload["notices"][0]["date"]  # 有日期
        assert payload["notices"][1]["text"] == "旧公告"  # 旧公告保留
        assert payload["title"] == "小涩猫咖啡厅"  # 全量写回，其余字段不动

    async def test_default_tag(self, site, admin):
        await tools.catcafe_post_notice.ainvoke({"text": "默认类型"})
        assert site["calls"][1]["payload"]["notices"][0]["tag"] == "公告"

    async def test_invalid_tag_aborts_before_http(self, site, admin):
        out = await tools.catcafe_post_notice.ainvoke(
            {"text": "x", "tag": "灌水"})
        assert "tag 只能是" in out
        assert site["calls"] == []  # 校验在 GET 之前，一次请求都不发

    async def test_empty_text_aborts(self, site, admin):
        out = await tools.catcafe_post_notice.ainvoke({"text": "  "})
        assert "空的" in out and site["calls"] == []

    async def test_non_admin_refused_no_http(self, site, non_admin):
        out = await tools.catcafe_post_notice.ainvoke({"text": "我要发公告"})
        assert "管理员" in out and site["calls"] == []

    async def test_not_saved_warns_recheck(self, site, admin, monkeypatch):
        async def weird(method, path, raw_body=None):
            if method == "GET":
                return {"notices": []}, None
            return {"status": "weird"}, None
        monkeypatch.setattr(tools, "_request", weird)
        out = await tools.catcafe_post_notice.ainvoke({"text": "x"})
        assert "重新读一遍" in out

    async def test_oversize_refused(self, site, admin, monkeypatch):
        monkeypatch.setattr(tools, "_MAX_PUT_BYTES", 10)
        out = await tools.catcafe_post_notice.ainvoke({"text": "x"})
        assert "1MB 上限" in out
        assert [c["method"] for c in site["calls"]] == ["GET"]  # 中止在 PUT 前


class TestFieldUpdates:
    async def test_set_slogan_only_field(self, site, admin):
        out = await tools.catcafe_set_slogan.ainvoke({"text": "新标语"})
        assert "完成" in out
        payload = site["calls"][1]["payload"]
        assert payload["slogan"] == "新标语"
        assert payload["authorStatus"] == "赶稿中"      # 其余字段原样
        assert payload["notices"][0]["text"] == "旧公告"

    async def test_set_status(self, site, admin):
        await tools.catcafe_set_status.ainvoke({"text": "休假中"})
        payload = site["calls"][1]["payload"]
        assert payload["authorStatus"] == "休假中"
        assert payload["slogan"] == "咖啡厅长期营业中~"

    async def test_set_slogan_non_admin(self, site, non_admin):
        out = await tools.catcafe_set_slogan.ainvoke({"text": "x"})
        assert "管理员" in out and site["calls"] == []


class TestListMessages:
    async def test_numbered_list_with_reply_marker(self, site):
        out = await tools.catcafe_list_messages.ainvoke({})
        assert "共 2 条" in out and "#0" in out and "#1" in out
        assert "路人甲" in out and "店长什么时候更新呀" in out
        assert "已回复（猪咪君君）：欢迎常来~" in out
        assert site["calls"][0]["path"] == "/api/messages"

    async def test_multiline_content_flattened(self, site):
        """content 里的换行在列表里压成空格（防模型抄行时错位）。"""
        out = await tools.catcafe_list_messages.ainvoke({})
        assert "打卡~ 今天也来撸猫" in out

    async def test_empty_board(self, site):
        site["msgs"].clear()
        out = await tools.catcafe_list_messages.ainvoke({})
        assert "空的" in out

    async def test_error_passthrough(self, monkeypatch):
        async def down(method, path, raw_body=None):
            return None, "连不上站点接口（ConnectError），稍后再试。"
        monkeypatch.setattr(tools, "_request", down)
        out = await tools.catcafe_list_messages.ainvoke({})
        assert "连不上站点接口" in out


class TestReplyMessage:
    async def test_reply_posts_exact_triple(self, site, admin):
        """接口无 id：nick+time+content 精确三元组定位，必须从列表原样取回——
        content 含换行的长文本绝不让模型转抄。"""
        out = await tools.catcafe_reply_message.ainvoke(
            {"index": 1, "text": "店长在赶稿，帮你转达~"})
        assert "已回复 #1（小猪）" in out
        assert [c["method"] for c in site["calls"]] == ["GET", "POST"]
        post = site["calls"][1]
        assert post["path"] == "/api/agent/messages/reply"
        src = site["msgs"][1]
        assert post["payload"]["nick"] == src["nick"]
        assert post["payload"]["time"] == src["time"]
        assert post["payload"]["content"] == src["content"]  # 含 \n 原样
        assert post["payload"]["reply"] == "店长在赶稿，帮你转达~"

    async def test_empty_text_refused_before_http(self, site, admin):
        """空回复 = 清除已有回复（2026-08-18 探测事故实锤），必须硬拒。"""
        out = await tools.catcafe_reply_message.ainvoke({"index": 0, "text": "  "})
        assert "不能为空" in out and site["calls"] == []

    async def test_index_out_of_range_no_post(self, site, admin):
        out = await tools.catcafe_reply_message.ainvoke({"index": 9, "text": "x"})
        assert "不存在" in out and "catcafe_list_messages" in out
        assert [c["method"] for c in site["calls"]] == ["GET"]  # 只拉了列表

    async def test_non_admin_no_http(self, site, non_admin):
        out = await tools.catcafe_reply_message.ainvoke({"index": 0, "text": "x"})
        assert "管理员" in out and site["calls"] == []

    async def test_not_ok_warns_recheck(self, site, admin, monkeypatch):
        async def weird(method, path, raw_body=None):
            if method == "GET":
                return [dict(site["msgs"][0])], None
            return {"status": "weird"}, None
        monkeypatch.setattr(tools, "_request", weird)
        out = await tools.catcafe_reply_message.ainvoke({"index": 0, "text": "x"})
        assert "重新读一遍" in out


class TestDeleteMessage:
    async def test_delete_posts_exact_triple(self, site, admin):
        out = await tools.catcafe_delete_message.ainvoke({"index": 0})
        assert "已删除 #0（路人甲）" in out and "记日志" in out
        post = site["calls"][1]
        assert post["path"] == "/api/agent/messages/delete"
        src = site["msgs"][0]
        assert post["payload"]["nick"] == src["nick"]
        assert post["payload"]["time"] == src["time"]
        assert post["payload"]["content"] == src["content"]
        assert "reason" not in post["payload"]  # 空 reason 不上送

    async def test_reason_sent_when_given(self, site, admin):
        await tools.catcafe_delete_message.ainvoke({"index": 0, "reason": "广告"})
        assert site["calls"][1]["payload"]["reason"] == "广告"

    async def test_expect_nick_match_ok(self, site, admin):
        out = await tools.catcafe_delete_message.ainvoke(
            {"index": 0, "expect_nick": "路人甲"})
        assert "已删除" in out

    async def test_expect_nick_mismatch_aborts_before_post(self, site, admin):
        """删除不可恢复：昵称复核对不上必须中止（防列表变动删错人）。"""
        out = await tools.catcafe_delete_message.ainvoke(
            {"index": 0, "expect_nick": "小猪"})
        assert "对不上" in out and "没动手" in out
        assert [c["method"] for c in site["calls"]] == ["GET"]

    async def test_index_out_of_range_no_post(self, site, admin):
        out = await tools.catcafe_delete_message.ainvoke({"index": -1})
        assert "不存在" in out
        assert [c["method"] for c in site["calls"]] == ["GET"]

    async def test_non_admin_no_http(self, site, non_admin):
        out = await tools.catcafe_delete_message.ainvoke({"index": 0})
        assert "管理员" in out and site["calls"] == []
