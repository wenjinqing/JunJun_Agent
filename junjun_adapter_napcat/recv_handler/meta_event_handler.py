"""元事件处理（心跳/生命周期）。阶段 1 仅记录。"""

from ..logger import logger


class MetaEventHandler:
    def __init__(self):
        pass

    async def handle_meta_event(self, raw_message: dict) -> None:
        # OneBot 11 字段为 meta_event_type（原 adapter 同名），不是 meta_type
        meta_type = raw_message.get("meta_event_type")
        if meta_type == "lifecycle":
            sub = raw_message.get("sub_type")
            if sub == "connect":
                logger.info("NapCat 已连接 (lifecycle connect)")
        elif meta_type == "heartbeat":
            logger.debug("NapCat heartbeat")


meta_event_handler = MetaEventHandler()
