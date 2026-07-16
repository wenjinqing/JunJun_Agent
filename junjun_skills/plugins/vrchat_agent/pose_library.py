"""预设动作库：把"挥手/点头"等语义动作翻译成 FrameState 时间序列。

这里的动作以关键帧 + 线性/正弦插值生成。每个 build_* 返回
list[(t_seconds, FrameState)]，喂给 AnyaDanceClient.play_sequence。
"""

from __future__ import annotations

import math
from typing import Callable

from .anya_client import (
    ControllerInput,
    DevicePose,
    FrameState,
    standing_frame_state,
    tpose_frame_state,
)


# 一致的小工具 --------------------------------------------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _swing(t: float, amp: float, freq: float, phase: float = 0.0) -> float:
    return amp * math.sin(2.0 * math.pi * freq * t + phase)


# 动作构建器 ----------------------------------------------------------------
def build_wave(duration: float = 3.0, rate: int = 60) -> list[tuple[float, FrameState]]:
    """右手举高并左右摆动挥手，左手自然下垂。"""
    base = standing_frame_state()
    frames: list[tuple[float, FrameState]] = []
    steps = int(duration * rate)
    for i in range(steps + 1):
        t = i / rate
        # 右手举到肩高以上，手腕左右摆
        swing = _swing(t, amp=0.18, freq=1.2)
        right = DevicePose(
            position=(0.35 + swing * 0.2, 1.55, 0.15),
            rotation=base.devices[2].rotation,
        )
        devs = list(base.devices)
        devs[2] = right
        frames.append((t, FrameState(devs, [ControllerInput(), ControllerInput()])))
    return frames


def build_nod(duration: float = 1.6, rate: int = 60) -> list[tuple[float, FrameState]]:
    """HMD 上下点头两次。"""
    base = standing_frame_state()
    frames: list[tuple[float, FrameState]] = []
    steps = int(duration * rate)
    for i in range(steps + 1):
        t = i / rate
        # 点头：HMD 绕 X 轴小幅俯仰
        pitch = _swing(t, amp=0.35, freq=1.25)
        hmd = DevicePose(
            position=base.devices[0].position,
            rotation=(math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2)),
        )
        devs = list(base.devices)
        devs[0] = hmd
        frames.append((t, FrameState(devs, [ControllerInput(), ControllerInput()])))
    return frames


def build_bow(duration: float = 2.0, rate: int = 60) -> list[tuple[float, FrameState]]:
    """鞠躬：HMD 与腰前倾再回正。"""
    base = standing_frame_state()
    frames: list[tuple[float, FrameState]] = []
    steps = int(duration * rate)
    for i in range(steps + 1):
        t = i / rate
        # 0->1->0 的钟形
        envelope = math.sin(math.pi * t / duration)
        pitch = 0.6 * envelope
        rot = (math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2))
        devs = list(base.devices)
        devs[0] = DevicePose(base.devices[0].position, rot)
        devs[3] = DevicePose(base.devices[3].position, rot)
        frames.append((t, FrameState(devs, [ControllerInput(), ControllerInput()])))
    return frames


def build_clap(duration: float = 2.4, rate: int = 60) -> list[tuple[float, FrameState]]:
    """拍手：两手在胸前反复合拢分开。"""
    base = standing_frame_state()
    frames: list[tuple[float, FrameState]] = []
    steps = int(duration * rate)
    for i in range(steps + 1):
        t = i / rate
        sep = 0.18 + 0.12 * (1.0 + math.cos(2 * math.pi * 1.6 * t)) / 2.0
        left = DevicePose((-sep, 1.30, 0.20), base.devices[1].rotation)
        right = DevicePose((sep, 1.30, 0.20), base.devices[2].rotation)
        devs = list(base.devices)
        devs[1] = left
        devs[2] = right
        frames.append((t, FrameState(devs, [ControllerInput(), ControllerInput()])))
    return frames


def build_jump(duration: float = 1.2, rate: int = 60) -> list[tuple[float, FrameState]]:
    """小跳：整体 Y 抛物线。用于测试全身位移。"""
    base = standing_frame_state()
    frames: list[tuple[float, FrameState]] = []
    steps = int(duration * rate)
    for i in range(steps + 1):
        t = i / rate
        h = 0.25 * math.sin(math.pi * t / duration)
        devs = [
            DevicePose((d.position[0], d.position[1] + h, d.position[2]), d.rotation)
            for d in base.devices
        ]
        frames.append((t, FrameState(devs, [ControllerInput(), ControllerInput()])))
    return frames


def build_walk_input(duration: float = 3.0, rate: int = 60, forward: bool = True) -> list[tuple[float, FrameState]]:
    """通过摇杆驱动 VRChat 内行走(不改变位姿，只改输入)。

    这是方案 3 里"自主在 VRChat 内移动"的最简形态：靠 AnyaDance 的
    inputs.joystick_y 推动角色前进，位姿由 VRChat 自己驱动。
    """
    base = standing_frame_state()
    frames: list[tuple[float, FrameState]] = []
    steps = int(duration * rate)
    joy_y = 1.0 if forward else -1.0
    for i in range(steps + 1):
        t = i / rate
        left = ControllerInput(joystick_y=joy_y)
        right = ControllerInput(joystick_y=joy_y)
        frames.append((t, FrameState(list(base.devices), [left, right])))
    return frames


# 动作登记表 ---------------------------------------------------------------
# 每项: name -> (描述, 构建器, 默认保持帧)
POSE_LIBRARY: dict[str, tuple[str, Callable[..., list[tuple[float, FrameState]]]]] = {
    "wave": ("举手挥手致意", build_wave),
    "nod": ("点头两次", build_nod),
    "bow": ("鞠躬", build_bow),
    "clap": ("拍手", build_clap),
    "jump": ("小跳一下", build_jump),
    "walk_forward": ("摇杆前进(在 VRChat 内行走)", build_walk_input),
    "tpose": ("T-pose 静态姿势", None),  # 静态，无构建器
    "standing": ("自然站立静态姿势", None),
}
