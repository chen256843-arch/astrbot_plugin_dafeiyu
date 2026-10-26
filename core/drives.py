"""驱力系统：六维连续状态 + 一阶惯性滤波，以及到五基态动作的映射。"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .collapse import Collapse, collapse
from .util import clamp

DRIVES = ("social", "curiosity", "play", "rest", "recognition", "achievement")
LABELS = {
    "social": "社交",
    "curiosity": "好奇",
    "play": "玩耍",
    "rest": "休息",
    "recognition": "认可",
    "achievement": "成就",
}
# 每小时自然上升速率
RATES = {
    "social": 3.0,
    "curiosity": 4.0,
    "play": 3.0,
    "rest": 2.5,
    "recognition": 1.5,
    "achievement": 1.0,
}
INERTIA = 0.7  # v_t = 0.7·v_{t-1} + 0.3·v_target
NAP_ON = 80.0
NAP_OFF = 25.0
REST_RECOVERY = 12.0  # 打盹/睡眠时每小时恢复

BASIS = ("读", "玩", "唤醒", "安静", "学")
TEMPERATURES = {"focused": 0.55, "normal": 1.0, "diffuse": 1.45}


@dataclass
class Drives:
    value: dict[str, float] = field(default_factory=lambda: {d: 40.0 for d in DRIVES})
    target: dict[str, float] = field(default_factory=lambda: {d: 40.0 for d in DRIVES})
    napping: bool = False
    mood: float = 0.2  # 心情效价 [-1, 1]

    # ---- 动力学 ----
    def tick(self, dt_hours: float, asleep: bool = False) -> None:
        for d in DRIVES:
            if d == "rest":
                if self.napping or asleep:
                    self.target[d] -= REST_RECOVERY * dt_hours
                else:
                    self.target[d] += RATES[d] * dt_hours
            else:
                rate = RATES[d] * (0.3 if asleep else 1.0)
                self.target[d] += rate * dt_hours
            self.target[d] = clamp(self.target[d], 0.0, 100.0)
            self.value[d] = clamp(INERTIA * self.value[d] + (1 - INERTIA) * self.target[d], 0.0, 100.0)
        if not self.napping and self.value["rest"] >= NAP_ON:
            self.napping = True
        elif self.napping and self.value["rest"] <= NAP_OFF:
            self.napping = False
        # 心情向基线回归
        self.mood += (0.2 - self.mood) * (1 - math.exp(-dt_hours / 6))

    def satisfy(self, drive: str, amount: float) -> None:
        """事件使驱力下降（被满足）。只改目标值，由惯性滤波平滑传到 value。"""
        self.target[drive] = clamp(self.target[drive] - amount, 0.0, 100.0)

    def bump(self, drive: str, amount: float) -> None:
        self.target[drive] = clamp(self.target[drive] + amount, 0.0, 100.0)

    def feel(self, valence: float, weight: float = 0.3) -> None:
        self.mood = clamp(self.mood + weight * valence, -1.0, 1.0)

    # ---- 决策接口 ----
    def basis_prefs(self) -> dict[str, float]:
        v = {d: self.value[d] / 100 for d in DRIVES}
        return {
            "读": 0.6 * v["curiosity"] + 0.3 * v["achievement"] + 0.05,
            "玩": 0.8 * v["play"] + 0.2 * v["social"] + 0.05,
            "唤醒": 0.7 * v["social"] + 0.3 * v["recognition"] + 0.02,
            "安静": 0.9 * v["rest"] + 0.05,
            "学": 0.5 * v["achievement"] + 0.4 * v["curiosity"] + 0.1 * v["recognition"] + 0.05,
        }

    def decide(self, hour: float, rng: random.Random | None = None) -> Collapse:
        """五基态决策：驱力振幅 + 情境（时段）振幅与相位。"""
        night = hour >= 22 or hour < 6
        if night:
            ctx = {"安静": 0.6, "读": 0.1}
            phase = {"安静": 0.0, "玩": math.pi, "唤醒": math.pi}
        elif 6 <= hour < 12:
            ctx = {"学": 0.3, "读": 0.2}
            phase = {"学": 0.0, "读": 0.0, "安静": math.pi / 2}
        elif 12 <= hour < 18:
            ctx = {"玩": 0.3, "唤醒": 0.1}
            phase = {"玩": 0.0, "唤醒": 0.0}
        else:
            ctx = {"读": 0.2, "唤醒": 0.2, "安静": 0.1}
            phase = {}
        return collapse(self.basis_prefs(), rng, context=ctx, phase=phase)

    def recall_temperature(self, rng: random.Random | None = None) -> tuple[str, float]:
        v = {d: self.value[d] / 100 for d in DRIVES}
        prefs = {
            "focused": 0.5 * v["achievement"] + 0.3 * (1 - v["rest"]) + 0.05,
            "normal": 0.35,
            "diffuse": 0.4 * v["play"] + 0.3 * v["curiosity"] + 0.3 * v["rest"] + 0.05,
        }
        label = collapse(prefs, rng).choice
        return label, TEMPERATURES[label]

    def drive_bias(self, node_drive: str | None) -> float:
        """记忆与当前驱力相关时提权，范围约 0.85–1.15。"""
        if not node_drive or node_drive not in self.value:
            return 1.0
        return 1.0 + 0.3 * (self.value[node_drive] / 100 - 0.5)

    def dominant(self) -> str:
        return max(DRIVES, key=lambda d: self.value[d])

    def summary(self) -> str:
        return " ".join(f"{LABELS[d]}{self.value[d]:.0f}" for d in DRIVES)

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {"value": self.value, "target": self.target, "napping": self.napping, "mood": self.mood}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Drives":
        obj = cls()
        if d:
            obj.value.update(d.get("value", {}))
            obj.target.update(d.get("target", {}))
            obj.napping = bool(d.get("napping", False))
            obj.mood = float(d.get("mood", 0.2))
        return obj
