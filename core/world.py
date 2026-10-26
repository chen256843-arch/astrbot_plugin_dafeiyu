"""具身模拟环境：室内房间层 + 岛面层并存。

室内层：BFS 最短路；门 open/closed/locked 控制通行、视线与声传播；光按窗距指数衰减；
        温度在相邻房间扩散；灰尘随时间累积。
岛面层：地标有坐标；移动代价 ∝ 距离；其他智能体（猫、鸟）按速度逐时推进。
"""
from __future__ import annotations

import math
import random
from collections import deque

from .collapse import collapse, pick
from .util import clamp, dist

# ---------------------------------------------------------------- 室内
ROOMS: dict[str, float] = {  # 房间 → 到窗户的距离
    "卧室": 1.0,
    "客厅": 1.5,
    "厨房": 2.0,
    "书房": 1.0,
    "浴室": 3.0,
    "门厅": 4.0,
}
DEFAULT_DOORS = [
    ["卧室", "客厅", "open"],
    ["书房", "客厅", "open"],
    ["厨房", "客厅", "open"],
    ["浴室", "卧室", "closed"],
    ["门厅", "客厅", "open"],
    ["门厅", "户外", "closed"],
]
LIGHT_LAMBDA = 2.5
SOUND_ATTEN = {"open": 0.6, "closed": 0.25, "locked": 0.2}
HEAT_K = {"open": 0.25, "closed": 0.05, "locked": 0.05}

# ---------------------------------------------------------------- 岛面
LANDMARKS: dict[str, tuple[float, float]] = {
    "屋子": (50.0, 50.0),
    "菜棚": (68.0, 44.0),
    "阳台": (44.0, 38.0),
    "水井": (38.0, 62.0),
    "老树": (70.0, 66.0),
    "崖边": (20.0, 80.0),
    "码头": (30.0, 22.0),
}
MOVE_COST = 0.08  # 每单位距离消耗的休息驱力


class World:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.doors: list[list[str]] = [list(d) for d in DEFAULT_DOORS]
        self.rooms: dict[str, dict] = {r: {"temp": 22.0, "dust": 10.0} for r in ROOMS}
        self.her = {"room": "客厅", "pos": list(LANDMARKS["屋子"]), "target": None, "speed": 30.0}
        self.agents: dict[str, dict] = {
            "小灰": {"kind": "cat", "pos": [58.0, 34.0], "target": None, "speed": 40.0, "hunger": 30.0, "intimacy": 30.0, "mode": "游荡"},
            "鸟": {"kind": "bird", "pos": list(LANDMARKS["老树"]), "target": None, "speed": 120.0, "mode": "回树"},
        }

    # ------------------------------------------------ 室内图
    def _neighbors(self, room: str, passable_only: bool = True) -> list[tuple[str, str]]:
        out = []
        for a, b, state in self.doors:
            if passable_only and state == "locked":
                continue
            if a == room:
                out.append((b, state))
            elif b == room:
                out.append((a, state))
        return out

    def path(self, src: str, dst: str) -> list[str] | None:
        """BFS 最短路，locked 门不可通行。"""
        if src == dst:
            return [src]
        prev: dict[str, str | None] = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt, _ in self._neighbors(cur):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == dst:
                    route = [dst]
                    while prev[route[-1]] is not None:
                        route.append(prev[route[-1]])
                    return route[::-1]
                q.append(nxt)
        return None

    def can_see(self, a: str, b: str) -> bool:
        return a == b or any(n == b and s == "open" for n, s in self._neighbors(a, passable_only=False))

    def sound_level(self, src: str, dst: str, loudness: float = 1.0) -> float:
        """声音沿门传播，每过一扇门按门状态衰减，取最响路径。"""
        best = {src: loudness}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt, state in self._neighbors(cur, passable_only=False):
                lvl = best[cur] * SOUND_ATTEN[state]
                if lvl > best.get(nxt, 0.0) + 1e-9:
                    best[nxt] = lvl
                    q.append(nxt)
        return best.get(dst, 0.0)

    def set_door(self, a: str, b: str, state: str) -> bool:
        for d in self.doors:
            if {d[0], d[1]} == {a, b}:
                d[2] = state
                return True
        return False

    def light(self, room: str, irradiance: float, lamp: float = 0.0) -> float:
        return clamp(irradiance * math.exp(-ROOMS.get(room, 3.0) / LIGHT_LAMBDA) + lamp, 0.0, 1.0)

    def tick_rooms(self, dt_h: float, outdoor_temp: float, irradiance: float, warmth: float = 0.0) -> None:
        temps = {r: s["temp"] for r, s in self.rooms.items()}
        delta = {r: 0.0 for r in temps}
        for a, b, state in self.doors:
            if a in temps and b in temps:
                flux = HEAT_K[state] * (temps[b] - temps[a]) * dt_h
                delta[a] += flux
                delta[b] -= flux
        for r, s in self.rooms.items():
            delta[r] += 0.05 * (outdoor_temp - temps[r]) * dt_h  # 与室外交换
            delta[r] += 1.5 * irradiance * math.exp(-ROOMS[r] / LIGHT_LAMBDA) * dt_h  # 日照升温
            if warmth and r in ("客厅", "卧室"):
                delta[r] += 0.2 * max(0.0, 22 + warmth - temps[r]) * dt_h
            s["temp"] = round(temps[r] + delta[r], 2)
            s["dust"] = clamp(s["dust"] + 0.4 * dt_h, 0.0, 100.0)

    def clean(self, room: str) -> float:
        d = self.rooms[room]["dust"]
        self.rooms[room]["dust"] = 0.0
        return d

    def dustiest(self) -> str:
        return max(self.rooms, key=lambda r: self.rooms[r]["dust"])

    # ------------------------------------------------ 岛面
    def her_location(self) -> str:
        pos = tuple(self.her["pos"])
        name, d = min(((n, dist(pos, p)) for n, p in LANDMARKS.items()), key=lambda x: x[1])
        return name if d < 4 else "路上"

    def indoors(self) -> bool:
        return self.her_location() == "屋子" and self.her["target"] is None

    def go(self, landmark: str) -> float:
        """设定目的地，返回移动代价（∝ 距离）。"""
        if landmark not in LANDMARKS:
            raise KeyError(landmark)
        d = dist(tuple(self.her["pos"]), LANDMARKS[landmark])
        self.her["target"] = list(LANDMARKS[landmark]) if d > 0.5 else None
        return d * MOVE_COST

    def go_room(self, room: str) -> list[str] | None:
        if self.her_location() != "屋子":
            self.go("屋子")
        route = self.path(self.her["room"], room)
        if route:
            self.her["room"] = room
        return route

    @staticmethod
    def _advance(agent: dict, dt_h: float) -> bool:
        """沿直线向目标推进，到达返回 True。"""
        tgt = agent.get("target")
        if not tgt:
            return False
        pos = agent["pos"]
        d = dist(tuple(pos), tuple(tgt))
        step = agent["speed"] * dt_h
        if step >= d:
            agent["pos"] = list(tgt)
            agent["target"] = None
            return True
        pos[0] += (tgt[0] - pos[0]) * step / d
        pos[1] += (tgt[1] - pos[1]) * step / d
        return False

    def tick_agents(self, dt_h: float, daylight: bool) -> list[str]:
        events: list[str] = []
        if self._advance(self.her, dt_h):
            events.append(f"她走到了{self.her_location()}")
        her_pos = tuple(self.her["pos"])

        cat = self.agents["小灰"]
        cat["hunger"] = clamp(cat["hunger"] + 5 * dt_h, 0, 100)
        near = dist(tuple(cat["pos"]), her_pos) < 15
        if cat["target"] is None or cat["mode"] == "跟随":
            mode = collapse(
                {
                    "去菜棚": 0.1 + cat["hunger"] / 100,
                    "跟随": 0.05 + (0.6 + cat["intimacy"] / 200 if near else 0.0),
                    "游荡": 0.35,
                },
                self.rng,
            ).choice
            if mode != cat["mode"]:
                events.append(f"小灰{mode}")
            cat["mode"] = mode
            if mode == "去菜棚":
                cat["target"] = list(LANDMARKS["菜棚"])
            elif mode == "跟随":
                cat["target"] = [her_pos[0] + self.rng.uniform(-2, 2), her_pos[1] + self.rng.uniform(-2, 2)]
            else:
                cat["target"] = list(LANDMARKS[pick({k: 1.0 for k in LANDMARKS}, self.rng)])
        self._advance(cat, dt_h)

        bird = self.agents["鸟"]
        want = "啄菜" if daylight else "回树"
        if bird["mode"] != want:
            bird["mode"] = want
            bird["target"] = list(LANDMARKS["菜棚" if daylight else "老树"])
            events.append("鸟飞去菜棚啄菜" if daylight else "鸟回到老树上")
        self._advance(bird, dt_h)
        return events

    def at(self, agent: str, landmark: str, radius: float = 4.0) -> bool:
        return dist(tuple(self.agents[agent]["pos"]), LANDMARKS[landmark]) < radius

    def render_map(self, w: int = 25, h: int = 13) -> str:
        grid = [["．"] * w for _ in range(h)]

        def put(p, ch):
            x = int(clamp(p[0] / 100 * (w - 1), 0, w - 1))
            y = int(clamp(p[1] / 100 * (h - 1), 0, h - 1))
            grid[y][x] = ch

        for name, p in LANDMARKS.items():
            put(p, name[0])
        put(self.agents["鸟"]["pos"], "鸟")
        put(self.agents["小灰"]["pos"], "猫")
        put(self.her["pos"], "她")
        legend = " ".join(f"{n[0]}={n}" for n in LANDMARKS)
        return "\n".join("".join(r) for r in grid) + "\n" + legend

    # ------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        return {"doors": self.doors, "rooms": self.rooms, "her": self.her, "agents": self.agents}

    @classmethod
    def from_dict(cls, d: dict | None, rng: random.Random | None = None) -> "World":
        w = cls(rng)
        if d:
            w.doors = d.get("doors", w.doors)
            w.rooms.update(d.get("rooms", {}))
            w.her.update(d.get("her", {}))
            for k, v in d.get("agents", {}).items():
                w.agents.setdefault(k, {}).update(v)
        return w
