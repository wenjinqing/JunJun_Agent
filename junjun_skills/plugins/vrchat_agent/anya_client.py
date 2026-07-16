"""AnyaDance UDP 协议客户端。

AnyaDance 是 SteamVR 虚拟设备驱动 + 伴随 UI，它从 127.0.0.1:39570 接收
60Hz 的 UDP JSON 帧，每个帧描述 6 个虚拟设备(HMD、左右手柄、腰、左右脚)
的位姿与控制器输入。本模块在 Python 侧复现该协议，让 MaiBot 能直接驱动
AnyaDance，从而让君君的虚拟形象在 VRChat 里活动。

协议字段来自 AnyaDance/src/core/protocol.cpp::SerializeFrame 与
frame_state.cpp 的预设位姿。保持与 v1 协议一致。
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ---- 协议常量(与 AnyaDance src/core/constants.h 对齐) -----------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 39570
PROTOCOL_VERSION = 1
STREAM_RATE_HZ = 60
MAX_DEVICE_Y = 2.0
RESET_HMD_Y = 1.50

# 设备 id 顺序与 AnyaDance kDevices 一致
DEVICE_IDS = (
    "hmd",
    "left_controller",
    "right_controller",
    "hip",
    "left_foot",
    "right_foot",
)


def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def _from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad * 0.5
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# 左右手柄的规范姿态(手心朝内、拇指朝前)，来自 tpose.h 的
# kLeftControllerCanonicalRotation / kRightControllerCanonicalRotation。
_LEFT_CANON = _quat_normalize((0.0, 0.0, -0.7071067811865475, 0.7071067811865475))
_RIGHT_CANON = _quat_normalize((0.0, 0.0, 0.7071067811865475, 0.7071067811865475))


@dataclass
class DevicePose:
    """单个设备的位姿。position 单位米，rotation 为 xyzw 四元数。"""

    position: tuple[float, float, float] = (0.0, 1.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    valid: bool = True
    connected: bool = True

    def to_dict(self) -> dict:
        x, y, z = self.position
        if y > MAX_DEVICE_Y:
            y = MAX_DEVICE_Y
        rx, ry, rz, rw = _quat_normalize(self.rotation)
        return {
            "valid": self.valid,
            "connected": self.connected,
            "pose": {
                "position": [x, y, z],
                "rotation_xyzw": [rx, ry, rz, rw],
            },
        }


@dataclass
class ControllerInput:
    """控制器输入：摇杆/按键/扳机。VRChat 用这些做移动与交互。"""

    trigger_click: bool = False
    trigger_value: float = 0.0
    menu_click: bool = False
    system_click: bool = False
    a_click: bool = False
    b_click: bool = False
    grip_click: bool = False
    grip_value: float = 0.0
    joystick_x: float = 0.0
    joystick_y: float = 0.0
    trackpad_x: float = 0.0
    trackpad_y: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trigger_click": self.trigger_click,
            "trigger_value": max(0.0, min(1.0, self.trigger_value)),
            "menu_click": self.menu_click,
            "system_click": self.system_click,
            "a_click": self.a_click,
            "b_click": self.b_click,
            "grip_click": self.grip_click,
            "grip_value": max(0.0, min(1.0, self.grip_value)),
            "joystick_x": max(-1.0, min(1.0, self.joystick_x)),
            "joystick_y": max(-1.0, min(1.0, self.joystick_y)),
            "trackpad_x": max(-1.0, min(1.0, self.trackpad_x)),
            "trackpad_y": max(-1.0, min(1.0, self.trackpad_y)),
        }


@dataclass
class FrameState:
    """完整的一帧：6 个设备位姿 + 左右控制器输入。"""

    devices: list[DevicePose] = field(default_factory=list)
    controllers: list[ControllerInput] = field(default_factory=lambda: [ControllerInput(), ControllerInput()])

    def __post_init__(self) -> None:
        if not self.devices:
            self.devices = neutral_frame_state().devices

    def to_packet(self) -> bytes:
        devices_obj = {did: dev.to_dict() for did, dev in zip(DEVICE_IDS, self.devices)}
        inputs_obj = {
            "left_controller": self.controllers[0].to_dict(),
            "right_controller": self.controllers[1].to_dict(),
        }
        payload = {"version": PROTOCOL_VERSION, "devices": devices_obj, "inputs": inputs_obj}
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def neutral_frame_state() -> FrameState:
    """MakeNeutralFrame 的 Python 复现。来自 frame_state.cpp。"""
    return FrameState(
        devices=[
            DevicePose((0.0, 1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((-0.45, 1.15, 0.0), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((0.45, 1.15, 0.0), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((0.0, 0.85, 0.0), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((-0.12, -0.01, 0.0), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((0.12, -0.01, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ]
    )


def tpose_frame_state() -> FrameState:
    """BuildResetTPose 的 Python 复现。来自 tpose.cpp / tpose.h。"""
    hmd_pos = (0.0, RESET_HMD_Y, 0.0)
    yaw = _from_yaw(0.0)
    left = _quat_mul(yaw, _LEFT_CANON)
    right = _quat_mul(yaw, _RIGHT_CANON)
    return FrameState(
        devices=[
            DevicePose(hmd_pos, yaw),
            DevicePose((-0.62, 1.33, -0.10), left),
            DevicePose((0.62, 1.33, -0.10), right),
            DevicePose((0.0, 1.07, -0.05), yaw),
            DevicePose((-0.09, 0.26, 0.10), yaw),
            DevicePose((0.09, 0.26, 0.10), yaw),
        ]
    )


def standing_frame_state() -> FrameState:
    """MakeStandingPose 的 Python 复现。手臂自然下垂。"""
    left_rot = _quat_normalize((-0.21, 0.09, -0.05, 0.97))
    right_rot = _quat_normalize((-0.21, -0.09, 0.05, 0.97))
    return FrameState(
        devices=[
            DevicePose((0.0, 1.50, 0.0), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((-0.18, 0.83, -0.10), left_rot),
            DevicePose((0.18, 0.83, -0.10), right_rot),
            DevicePose((0.0, 1.07, -0.05), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((-0.09, 0.26, 0.10), (0.0, 0.0, 0.0, 1.0)),
            DevicePose((0.09, 0.26, 0.10), (0.0, 0.0, 0.0, 1.0)),
        ]
    )


class AnyaDanceClient:
    """后台流送线程，按 60Hz 向 AnyaDance 发送当前帧。

    用法：调用 ``start()`` 启动后台线程，之后通过 ``play_sequence`` /
    ``hold`` / ``stop_motion`` 等方法下发动动作意图，由本类完成帧级翻译。
    所有方法都是线程安全的。
    """

    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT, rate_hz: int = STREAM_RATE_HZ) -> None:
        self._host = host
        self._port = port
        self._period = 1.0 / rate_hz
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.RLock()
        # 当前帧序列(time, FrameState)列表；播完后回到 hold_frame
        self._sequence: list[tuple[float, FrameState]] = []
        self._sequence_start: float = 0.0
        # 当前持续保持的帧(无序列播放时使用)
        self._hold_frame: FrameState = standing_frame_state()
        # 是否处于"活动"状态：True=持续流送，False=停止(发完当前序列就停)
        self._active = False

    # ---- 生命周期 ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="AnyaDanceClient", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ---- 主循环 -----------------------------------------------------------
    def _loop(self) -> None:
        next_t = time.monotonic()
        while self._running.is_set():
            frame = self._resolve_frame(time.monotonic())
            if frame is not None:
                try:
                    self._sock.sendto(frame.to_packet(), (self._host, self._port))
                except OSError:
                    pass
            next_t += self._period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _resolve_frame(self, now: float) -> Optional[FrameState]:
        with self._lock:
            if self._sequence:
                elapsed = now - self._sequence_start
                # 取序列中时间 <= elapsed 的最后一帧
                chosen = self._sequence[0][1]
                for t, f in self._sequence:
                    if t <= elapsed:
                        chosen = f
                    else:
                        break
                if elapsed >= self._sequence[-1][0]:
                    self._sequence = []
                return chosen
            return self._hold_frame if self._active else None

    # ---- 动作意图 API ------------------------------------------------------
    def hold(self, frame: FrameState) -> None:
        """让形象持续保持某个帧(用于站立、T-pose 等静态姿势)。"""
        with self._lock:
            self._sequence = []
            self._hold_frame = frame
            self._active = True

    def play_sequence(self, frames: list[tuple[float, FrameState]], then_hold: Optional[FrameState] = None) -> None:
        """播放一个时间轴帧序列。每项为 (相对秒, 帧状态)。

        播放结束后若给了 then_hold 则保持该帧，否则回到之前的 hold_frame。
        """
        if not frames:
            return
        with self._lock:
            self._sequence = list(frames)
            self._sequence_start = time.monotonic()
            self._active = True
            if then_hold is not None:
                self._hold_frame = then_hold

    def stop_motion(self) -> None:
        """停止一切动作流送，回到静默(形象会停在 AnyaDance 最后收到的帧)。"""
        with self._lock:
            self._sequence = []
            self._active = False

    def is_active(self) -> bool:
        with self._lock:
            return self._active or bool(self._sequence)

    def current_description(self) -> str:
        with self._lock:
            if self._sequence:
                return f"播放序列中({len(self._sequence)}帧剩余)"
            if self._active:
                return "保持静态姿势"
            return "停止"


# 进程级单例：插件内多个工具共享同一个流送线程
_client: Optional[AnyaDanceClient] = None
_client_lock = threading.Lock()


def get_client() -> AnyaDanceClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = AnyaDanceClient()
            _client.start()
        return _client
