import asyncio
from maim_message import (
    Router, RouteConfig, TargetConfig,
    MessageBase, Seg, UserInfo, GroupInfo, BaseMessageInfo, FormatInfo,
)

async def main():
    route = RouteConfig(route_config={
        "junjun": TargetConfig(url="ws://127.0.0.1:8092/ws", token=None)
    })
    router = Router(route)
    received = []
    async def on_msg(msg_dict):
        mb = MessageBase.from_dict(msg_dict)
        seg = mb.message_segment
        text = ""
        if seg.type == "seglist":
            for s in seg.data:
                if s.type == "text":
                    text += s.data
        elif seg.type == "text":
            text = seg.data
        print(f"[RECV] {text}", flush=True)
        received.append(text)
    router.register_class_handler(on_msg)
    run_task = asyncio.create_task(router.run())
    await asyncio.sleep(2)

    user_info = UserInfo(platform="junjun", user_id=12345, user_nickname="tester", user_cardname=None)
    group_info = GroupInfo(platform="junjun", group_id=999, group_name="test")
    msg_info = BaseMessageInfo(
        platform="junjun", message_id="1", time=0.0,
        user_info=user_info, group_info=group_info,
        template_info=None,
        format_info=FormatInfo(content_format=["text"], accept_format=["text"]),
        additional_config={},
    )
    seg = Seg(type="seglist", data=[Seg(type="text", data="hello junjun")])
    mb = MessageBase(message_info=msg_info, message_segment=seg, raw_message="hello junjun")
    ok = await router.send_message(mb)
    print(f"[SEND] ok={ok}", flush=True)
    await asyncio.sleep(3)
    print(f"[DONE] received {len(received)} replies", flush=True)
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
