"""junjun_cli 补丁面测试（0 token）：sink 捕获语义 / force_enable 只翻开关 /
capture_terminal 只收本会话 / patch_surface 替换后可用。"""

import asyncio
from types import SimpleNamespace

from scripts import junjun_cli as cli


def _seg(text: str):
    return SimpleNamespace(type="text", data=text)


class TestCliSink:
    def test_on_send_prints_and_flags_rejection(self, capsys):
        sink = cli.CliSink()
        ok = asyncio.run(sink.on_send(
            cli.CLI_CHAT_ID, [_seg("我琢磨了一下，这个我好像拆不动——再具体说说？")]))
        assert ok and sink.rejected
        assert "拆不动" in capsys.readouterr().out

    def test_other_chat_rejection_not_flagged(self):
        sink = cli.CliSink()
        asyncio.run(sink.on_send("qq:1:group", [_seg("拆不动")]))
        assert not sink.rejected   # 别的会话的交代不是本 CLI 的终态信号

    def test_normal_report_not_rejected(self):
        sink = cli.CliSink()
        asyncio.run(sink.on_send(cli.CLI_CHAT_ID, [_seg("弄好了：报告如下")]))
        assert not sink.rejected


class TestForceEnable:
    def test_overrides_switch_but_keeps_rest(self, monkeypatch):
        from junjun_agent.task_kernel import executor
        monkeypatch.setattr(executor, "_cfg",
                            lambda: {"enable": False, "max_steps": 6})
        cli.force_enable(executor)
        cfg = executor._cfg()
        assert cfg["enable"] is True
        assert cfg["max_steps"] == 6   # 其余配置照生产，不是 eval 那种整体替换


class TestCaptureTerminal:
    def test_captures_own_chat_only(self):
        sink = cli.CliSink()
        reported = []

        class FakeKernel:
            async def _report(self, plan):
                reported.append(plan.plan_id)

        fk = FakeKernel()
        cli.capture_terminal(fk, sink)
        own = SimpleNamespace(chat_id=cli.CLI_CHAT_ID, plan_id="p1")
        other = SimpleNamespace(chat_id="qq:1:group", plan_id="p2")
        asyncio.run(fk._report(own))
        asyncio.run(fk._report(other))
        assert list(sink.completed) == ["p1"]   # 别的会话的汇报不混进来
        assert reported == ["p1", "p2"]          # 原汇报两条都照常走


class TestPatchSurface:
    def test_replacements_callable_and_restorable(self, capsys):
        import junjun_agent.outbound as outbound
        import junjun_core.security as sec
        from junjun_agent.tasks import task_manager
        real = (outbound.send_proactive, sec.notify_admin,
                task_manager._record_outcome)
        try:
            sink = cli.CliSink()
            cli.patch_surface(sink)
            # 绑定方法每次取都是新对象，不能 is 比——按归属+行为断言
            assert outbound.send_proactive.__self__ is sink
            assert sec.notify_admin.__self__ is sink
            assert asyncio.run(outbound.send_proactive(cli.CLI_CHAT_ID, [_seg("hi")]))
            assert asyncio.run(sec.notify_admin("审批测试"))
            task_manager._record_outcome(cli.CLI_CHAT_ID, "task_kernel", "done", "x")
            assert "hi" in capsys.readouterr().out
        finally:
            (outbound.send_proactive, sec.notify_admin,
             task_manager._record_outcome) = real
