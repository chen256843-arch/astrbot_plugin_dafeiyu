"""生态与农业：剩饭 → 虫 → 猫捕食；鸟啄菜；4 块地；阳台 4 盆花；食材与菜谱。"""
from __future__ import annotations

import random

from .collapse import collapse, pick
from .util import clamp

CROPS: dict[str, dict] = {
    "番茄": {"days": 6, "opt": (18, 30), "price": 12, "seed": 3},
    "青菜": {"days": 3, "opt": (10, 25), "price": 6, "seed": 1},
    "萝卜": {"days": 5, "opt": (8, 22), "price": 8, "seed": 2},
    "草莓": {"days": 8, "opt": (15, 26), "price": 20, "seed": 5},
}
VARIETIES = {"普通": 1.0, "耐旱": 0.6, "抗病": 0.5}  # 耐旱：蒸发系数；抗病：病害系数
FLOWERS = ("茉莉", "月季", "向日葵", "薰衣草")
RECIPES: dict[str, dict[str, int]] = {
    "番茄炒蛋": {"番茄": 1, "鸡蛋": 1},
    "清炒青菜": {"青菜": 1},
    "萝卜汤": {"萝卜": 1},
    "草莓甜点": {"草莓": 1},
    "白粥": {"米": 1},
    "青菜粥": {"米": 1, "青菜": 1},
}
TASTE = {"番茄炒蛋": "咸", "清炒青菜": "清淡", "萝卜汤": "清淡", "草莓甜点": "甜", "白粥": "清淡", "青菜粥": "清淡"}
SEASON_BUG = {0: 0.3, 1: 1.0, 2: 1.5, 3: 1.0}  # 冬春夏秋


def season_of(month: int) -> int:
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(month, 3)


class Ecology:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.plots: list[dict] = [self._empty_plot() for _ in range(4)]
        self.pots: list[dict] = [{"kind": k, "bloom": 30.0, "water": 60.0} for k in FLOWERS]
        self.pantry: dict[str, int] = {"米": 3, "鸡蛋": 2, "猫粮": 3}
        self.meals: list[dict] = []  # 已做好的餐 {"name", "ts"}
        self.leftovers = 0
        self.bugs = 0.0
        self.taste = "清淡"

    @staticmethod
    def _empty_plot() -> dict:
        return {"crop": None, "growth": 0.0, "water": 50.0, "health": 100.0, "variety": "普通"}

    # ------------------------------------------------ 农业
    def plant(self, idx: int, crop: str) -> dict:
        p = self.plots[idx]
        p.update(self._empty_plot(), crop=crop, water=p["water"])
        p["variety"] = pick({"普通": 0.6, "耐旱": 0.2, "抗病": 0.2}, self.rng)
        return p

    def water(self, idx: int | None = None, amount: float = 40.0) -> int:
        targets = range(4) if idx is None else [idx]
        for i in targets:
            self.plots[i]["water"] = clamp(self.plots[i]["water"] + amount, 0, 100)
        for pot in self.pots if idx is None else []:
            pot["water"] = clamp(pot["water"] + amount, 0, 100)
        return len(targets)

    def harvest(self, idx: int) -> tuple[str, int] | None:
        p = self.plots[idx]
        if not p["crop"] or p["growth"] < 100:
            return None
        n = round(2 + 3 * p["health"] / 100)
        crop = p["crop"]
        self.pantry[crop] = self.pantry.get(crop, 0) + n
        self.plots[idx] = self._empty_plot()
        return crop, n

    def tick_farm(self, dt_h: float, weather: dict, greenhouse: int = 0, growth_bonus: float = 0.0) -> list[str]:
        events: list[str] = []
        temp = weather.get("temp", 22)
        rain = weather.get("precipitation", 0)
        humid = weather.get("humidity", 60)
        wind = weather.get("wind", 5)
        shield = 0.3 * greenhouse if greenhouse < 2 else 0.5  # 伤害减免
        for i, p in enumerate(self.plots):
            if not p["crop"]:
                continue
            spec = CROPS[p["crop"]]
            evap = (0.5 + max(0, temp - 20) * 0.15 + wind * 0.02) * dt_h * (1 - 0.25 * greenhouse)
            if p["variety"] == "耐旱":
                evap *= VARIETIES["耐旱"]
            p["water"] = clamp(p["water"] + rain * 8 * dt_h * (1 - 0.3 * greenhouse) - evap, 0, 100)
            lo, hi = spec["opt"]
            off = max(0, lo - temp, temp - hi)
            temp_f = max(0.0, 1 - 0.1 * off)
            water_f = 1.0 if 30 <= p["water"] <= 90 else 0.5 if p["water"] > 10 else 0.0
            rate = 100 / (spec["days"] * 24) * temp_f * water_f * (p["health"] / 100) * (1 + growth_bonus)
            p["growth"] = clamp(p["growth"] + rate * dt_h, 0, 100)
            dmg = 0.0
            if p["water"] < 15:
                dmg += 2 * dt_h
            if off > 8:
                dmg += 1.5 * dt_h
            sick = 0.02 * dt_h * (VARIETIES["抗病"] if p["variety"] == "抗病" else 1.0)
            if humid > 85 and self.rng.random() < sick:
                dmg += 15
                events.append(f"{i + 1}号地的{p['crop']}生病了")
            if dmg:
                p["health"] = clamp(p["health"] - dmg * (1 - shield), 0, 100)
            else:
                p["health"] = clamp(p["health"] + 0.5 * dt_h, 0, 100)
            if p["health"] <= 0:
                events.append(f"{i + 1}号地的{p['crop']}枯死了")
                self.plots[i] = self._empty_plot()
            elif p["growth"] >= 100 and not p.get("ripe_noted"):
                p["ripe_noted"] = True
                events.append(f"{i + 1}号地的{p['crop']}熟了")
        return events

    def bird_peck(self, dt_h: float, greenhouse: int = 0) -> str | None:
        planted = [i for i, p in enumerate(self.plots) if p["crop"]]
        if not planted or self.rng.random() > 0.3 * dt_h * (1 - 0.35 * greenhouse):
            return None
        i = self.rng.choice(planted)
        self.plots[i]["growth"] = clamp(self.plots[i]["growth"] - 3, 0, 100)
        self.plots[i]["health"] = clamp(self.plots[i]["health"] - 2, 0, 100)
        return f"鸟啄了{i + 1}号地的{self.plots[i]['crop']}"

    # ------------------------------------------------ 花卉
    def tick_flowers(self, dt_h: float, irradiance: float) -> list[str]:
        events = []
        for pot in self.pots:
            pot["water"] = clamp(pot["water"] - 1.2 * dt_h, 0, 100)
            if pot["water"] > 20:
                was = pot["bloom"] >= 100
                pot["bloom"] = clamp(pot["bloom"] + (0.4 + 2.0 * irradiance) * dt_h, 0, 100)
                if pot["bloom"] >= 100 and not was:
                    events.append(f"阳台的{pot['kind']}开花了")
        return events

    def blooming(self) -> int:
        return sum(1 for p in self.pots if p["bloom"] >= 100)

    def cut_flower(self) -> str | None:
        for pot in self.pots:
            if pot["bloom"] >= 100:
                pot["bloom"] = 30.0
                return pot["kind"]
        return None

    # ------------------------------------------------ 生态链
    def tick_bugs(self, dt_h: float, month: int, weather: dict, at_sea: bool = True) -> None:
        f = SEASON_BUG[season_of(month)]
        f *= 1.3 if weather.get("humidity", 60) > 70 else 1.0
        f *= 0.7 if weather.get("precipitation", 0) > 0.2 else 1.0
        f *= 1.2 if at_sea else 1.0
        self.bugs = clamp(self.bugs + self.leftovers * 0.3 * f * dt_h - 0.05 * self.bugs * dt_h, 0, 50)

    def cat_hunt(self, cat: dict) -> str | None:
        if self.bugs >= 1 and self.rng.random() < 0.5:
            self.bugs -= 1
            cat["hunger"] = clamp(cat["hunger"] - 10, 0, 100)
            return "小灰在菜棚边扑到了一只虫"
        return None

    def small_animals(self, dt_h: float) -> str | None:
        if self.leftovers and self.rng.random() < 0.05 * dt_h:
            self.leftovers -= 1
            return "有小动物偷吃了一份剩饭"
        return None

    # ------------------------------------------------ 厨房
    def feasible(self) -> list[str]:
        return [r for r, need in RECIPES.items() if all(self.pantry.get(k, 0) >= v for k, v in need.items())]

    def daily_taste(self) -> str:
        self.taste = pick({"咸": 0.35, "甜": 0.25, "清淡": 0.4}, self.rng)
        return self.taste

    def cook(self, t: float) -> str | None:
        opts = self.feasible()
        if not opts:
            return None
        prefs = {r: 1.0 + (1.0 if TASTE[r] == self.taste else 0.0) for r in opts}
        dish = collapse(prefs, self.rng).choice
        for k, v in RECIPES[dish].items():
            self.pantry[k] -= v
        self.meals.extend({"name": dish, "ts": t} for _ in range(2))
        return dish

    def eat(self) -> str | None:
        if not self.meals:
            return None
        return self.meals.pop(0)["name"]

    def spoil(self, t: float, hours: float = 12) -> int:
        stale = [m for m in self.meals if t - m["ts"] > hours * 3600]
        self.meals = [m for m in self.meals if t - m["ts"] <= hours * 3600]
        self.leftovers += len(stale)
        return len(stale)

    # ------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("plots", "pots", "pantry", "meals", "leftovers", "bugs", "taste")}

    @classmethod
    def from_dict(cls, d: dict | None, rng: random.Random | None = None) -> "Ecology":
        e = cls(rng)
        for k, v in (d or {}).items():
            if hasattr(e, k):
                setattr(e, k, v)
        return e
