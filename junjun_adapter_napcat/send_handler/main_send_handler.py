"""发消息处理：网关回复 MessageBase -> NapCat OneBot API。"""

from typing import List

from maim_message import MessageBase, Seg

from ..logger import logger
from .nc_sending import nc_message_sender


class SendHandler:
    async def handle_message(self, raw_message_base_dict: dict) -> None:
        try:
            msg = MessageBase.from_dict(raw_message_base_dict)
        except Exception as e:
            logger.error(f"反序列化网关消息失败: {e}")
            return
        seg = msg.message_segment
        info = msg.message_info
        group_info = info.group_info

        # 合并转发不是普通消息段，走独立 OneBot action（send_group/private_forward_msg）
        seg, forwards = self._extract_forwards(seg)
        for nodes in forwards:
            if group_info:
                f_action, f_params = "send_group_forward_msg", {
                    "group_id": int(group_info.group_id), "messages": nodes}
            else:
                f_action, f_params = "send_private_forward_msg", {
                    "user_id": int(info.user_info.user_id), "messages": nodes}
            f_resp = await nc_message_sender.send_message_to_napcat(f_action, f_params)
            if f_resp.get("status") == "ok":
                logger.info(f"合并转发已发 [{f_action} {len(nodes)} 节点]")
            else:
                logger.warning(f"合并转发发送失败: {f_resp}")

        # poke 不是消息段，走独立 OneBot action（send_group_poke/send_private_poke）
        seg, pokes = self._extract_pokes(seg)
        for target in pokes:
            if group_info:
                p_action, p_params = "send_group_poke", {
                    "group_id": int(group_info.group_id), "user_id": int(target)}
            else:
                p_action, p_params = "send_private_poke", {"user_id": int(target)}
            p_resp = await nc_message_sender.send_message_to_napcat(p_action, p_params)
            if p_resp.get("status") == "ok":
                logger.info(f"戳一戳已发 [{p_action} target={target}]")
            else:
                logger.warning(f"戳一戳发送失败: {p_resp}")

        processed = await self._process_seg(seg)
        if not processed:
            if not pokes and not forwards:
                logger.warning("无有效发送内容")
            return

        if group_info:
            params = {
                "group_id": int(group_info.group_id),
                "message": processed,
            }
            action = "send_group_msg"
        else:
            params = {
                "user_id": int(info.user_info.user_id),
                "message": processed,
            }
            action = "send_private_msg"

        resp = await nc_message_sender.send_message_to_napcat(action, params)
        if resp.get("status") == "ok":
            logger.info(f"已发送到 NapCat [{action}]")
        else:
            logger.warning(f"NapCat 发送失败: {resp}")

    @staticmethod
    def _extract_pokes(seg: Seg) -> tuple:
        """把 poke 段从消息段中摘出来。返回 (剩余 seg, poke 目标 id 列表)。"""
        if seg.type == "seglist" and isinstance(seg.data, list):
            rest = [s for s in seg.data if s.type != "poke"]
            pokes = [str(s.data) for s in seg.data if s.type == "poke"]
            if not rest:
                return Seg(type="text", data=""), pokes
            return (Seg(type="seglist", data=rest) if len(rest) > 1 else rest[0]), pokes
        if seg.type == "poke":
            return Seg(type="text", data=""), [str(seg.data)]
        return seg, []

    async def _process_seg(self, seg: Seg) -> List[dict]:
        """Seg -> OneBot message 数组。"""
        payload: List[dict] = []
        if seg.type == "seglist" and isinstance(seg.data, list):
            for sub in seg.data:
                payload = self._process_one(sub, payload)
        else:
            payload = self._process_one(seg, payload)
        return payload

    @staticmethod
    def _extract_forwards(seg: Seg) -> tuple:
        """把 forward 段摘出来。data 为 JSON: [{"type":"node","data":{...}}, ...]。
        返回 (剩余 seg, [nodes_list, ...])。"""
        import json

        def _nodes(s):
            try:
                nodes = json.loads(s.data)
                return nodes if isinstance(nodes, list) else None
            except Exception:
                logger.warning(f"forward 段 JSON 解析失败: {str(s.data)[:80]}")
                return None

        if seg.type == "seglist" and isinstance(seg.data, list):
            rest, forwards = [], []
            for s in seg.data:
                if s.type == "forward":
                    nodes = _nodes(s)
                    if nodes:
                        forwards.append(nodes)
                else:
                    rest.append(s)
            if not rest:
                return Seg(type="text", data=""), forwards
            return (Seg(type="seglist", data=rest) if len(rest) > 1 else rest[0]), forwards
        if seg.type == "forward":
            nodes = _nodes(seg)
            return Seg(type="text", data=""), ([nodes] if nodes else [])
        return seg, []

    def _process_one(self, seg: Seg, payload: List[dict]) -> List[dict]:
        import json
        if seg.type == "text":
            text = seg.data
            if text:
                payload.append({"type": "text", "data": {"text": text}})
        elif seg.type == "image":
            payload.append({"type": "image", "data": {"file": seg.data}})
        elif seg.type in ("voice", "voiceurl"):
            payload.append({"type": "record", "data": {"file": seg.data}})
        elif seg.type in ("video", "videourl"):
            payload.append({"type": "video", "data": {"file": seg.data}})
        elif seg.type == "at":
            payload.append({"type": "at", "data": {"qq": str(seg.data)}})
        elif seg.type == "music":
            # JSON: {"type": "custom", "url": 卡片链接, "audio": 音频直链,
            #        "title": 曲名, "content": 歌手/说明, "image": 封面URL}
            try:
                data = json.loads(seg.data)
                data.setdefault("type", "custom")
                payload.append({"type": "music", "data": data})
            except Exception:
                logger.warning(f"music 段 JSON 解析失败: {str(seg.data)[:80]}")
        elif seg.type == "emoji":
            payload.append({"type": "face", "data": {"id": str(seg.data)}})
        elif seg.type == "reply":
            payload.append({"type": "reply", "data": {"id": str(seg.data)}})
        return payload


send_handler = SendHandler()
