"""经济：金币为单一记账单位；家具提供专门加成并有上限。"""
from __future__ import annotations

from .ecology import CROPS
from .util import now

INCOME = {"读书": 3, "家务": 5, "喂猫": 2, "剪花": 8}

GOODS: dict[str, int] = {"鸡蛋": 2, "米": 1, "猫粮": 3, **{f"{c}种子": v["seed"] for c, v in CROPS.items()}}

# 家具：价格、加成类型、单件加成、最多件数
FURNITURE: dict[str, dict] = {
    "台灯": {"price": 30, "bonus": "light", "value": 0.3, "max": 2},
    "暖炉": {"price": 60, "bonus": "warmth", "value": 3.0, "max": 2},
    "香薰": {"price": 25, "bonus": "deodor", "value": 0.3, "max": 2},
    "猫爬架": {"price": 50, "bonus": "cat", "value": 0.5, "max": 1},
    "生长灯": {"price": 80, "bonus": "growth", "value": 0.15, "max": 2},
    "书架": {"price": 40, "bonus": "read", "value": 1.0, "max": 2},
    "温室棚": {"price": 100, "bonus": "greenhouse", "value": 1.0, "max": 2},
}
BONUS_CAP = {"light": 0.6, "warmth": 6.0, "deodor": 0.6, "cat": 0.5, "growth": 0.3, "read": 2.0, "greenhouse": 2.0}


class Economy:
    def __init__(self, coins: int = 50):
        self.coins = coins
        self.furniture: dict[str, int] = {}
        self.ledger: list[dict] = []

    def _log(self, amount: int, reason: str) -> None:
        self.ledger.append({"ts": now(), "amount": amount, "reason": reason, "balance": self.coins})
        self.ledger = self.ledger[-100:]

    def earn(self, amount: int, reason: str) -> int:
        self.coins += amount
        self._log(amount, reason)
        return self.coins

    def spend(self, amount: int, reason: str) -> bool:
        if amount > self.coins:
            return False
        self.coins -= amount
        self._log(-amount, reason)
        return True

    def bonus(self, kind: str) -> float:
        total = sum(FURNITURE[f]["value"] * n for f, n in self.furniture.items() if FURNITURE[f]["bonus"] == kind)
        return min(total, BONUS_CAP.get(kind, total))

    def buy_furniture(self, name: str) -> str:
        spec = FURNITURE.get(name)
        if not spec:
            return f"没有「{name}」这种家具"
        if self.furniture.get(name, 0) >= spec["max"]:
            return f"{name}已经到上限了"
        if not self.spend(spec["price"], f"买{name}"):
            return f"金币不够（{name} 要 {spec['price']}，只有 {self.coins}）"
        self.furniture[name] = self.furniture.get(name, 0) + 1
        return f"买了{name}，剩 {self.coins} 金币"

    def buy_goods(self, name: str, qty: int, pantry: dict[str, int]) -> str:
        price = GOODS.get(name)
        if price is None:
            return f"商店没有「{name}」"
        if not self.spend(price * qty, f"买{name}×{qty}"):
            return f"金币不够（需要 {price * qty}，只有 {self.coins}）"
        pantry[name] = pantry.get(name, 0) + qty
        return f"买了{name}×{qty}，剩 {self.coins} 金币"

    def catalog(self) -> str:
        goods = " ".join(f"{k}{v}" for k, v in GOODS.items())
        furn = " ".join(f"{k}{v['price']}({self.furniture.get(k, 0)}/{v['max']})" for k, v in FURNITURE.items())
        return f"食材/种子：{goods}\n家具：{furn}"

    def to_dict(self) -> dict:
        return {"coins": self.coins, "furniture": self.furniture, "ledger": self.ledger}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Economy":
        e = cls()
        if d:
            e.coins = d.get("coins", e.coins)
            e.furniture = d.get("furniture", {})
            e.ledger = d.get("ledger", [])
        return e
