"""对话接口 + 成本记账（日耗、剩余可用天数）。

两种后端：直接调用 OpenAI 兼容接口（DeepSeek 等），或由宿主框架注入的 backend
（例如 AstrBot 里已经配好的提供商）。backend 签名：
    async (messages, max_tokens, temperature) -> (text | None, 输入 token, 输出 token)
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

log = logging.getLogger("dafeiyu.llm")

Backend = Callable[[list[dict], int, float], Awaitable[tuple[str | None, int, int]]]


def today() -> str:
    return time.strftime("%Y-%m-%d")


class LLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        price_in: float = 0.002,
        price_out: float = 0.008,
        backend: Backend | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.price_in = price_in  # 每千 token 价格（元）
        self.price_out = price_out
        self.backend = backend
        self.usage: dict[str, dict] = {}
        self.balance: dict = {}  # {"value": 余额, "ts": 记录时间, "day": 日期}
        self.rule_actions = 0
        self.llm_actions = 0

    @property
    def enabled(self) -> bool:
        return bool(self.backend or self.api_key)

    def _record(self, tin: int, tout: int) -> None:
        u = self.usage.setdefault(today(), {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        u["calls"] += 1
        u["in"] += tin
        u["out"] += tout
        u["cost"] = round(u["cost"] + tin / 1000 * self.price_in + tout / 1000 * self.price_out, 5)
        if len(self.usage) > 60:
            for k in sorted(self.usage)[:-60]:
                del self.usage[k]

    async def chat(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.8) -> str | None:
        if self.backend:
            try:
                text, tin, tout = await self.backend(messages, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM 调用失败: %s", e)
                return None
            self._record(tin, tout)
            return (text or "").strip() or None
        if not self.api_key:
            return None
        import httpx

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 调用失败: %s", e)
            return None
        usage = data.get("usage") or {}
        self._record(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        return (data["choices"][0]["message"].get("content") or "").strip() or None

    def daily_burn(self, days: int = 7) -> float:
        recent = [self.usage[k]["cost"] for k in sorted(self.usage)[-days:]]
        return sum(recent) / len(recent) if recent else 0.0

    def report(self) -> str:
        u = self.usage.get(today(), {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        burn = self.daily_burn()
        total = self.rule_actions + self.llm_actions
        ratio = self.rule_actions / total * 100 if total else 0.0
        lines = [
            f"今日调用 {u['calls']} 次，输入 {u['in']} / 输出 {u['out']} token，约 ¥{u['cost']:.4f}",
            f"近 7 日边际日耗约 ¥{burn:.4f}",
            f"空间动作规则引擎占比 {ratio:.0f}%（{self.rule_actions}/{total}）",
        ]
        if self.balance:
            left = self.balance["value"] / burn if burn > 0 else float("inf")
            lines.append(f"账户余额 ¥{self.balance['value']:.2f}（{self.balance['day']} 记录），剩余可用约 {left:.0f} 天")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"usage": self.usage, "balance": self.balance, "rule": self.rule_actions, "llm": self.llm_actions}

    def load(self, d: dict | None) -> None:
        if d:
            self.usage = d.get("usage", {})
            self.balance = d.get("balance", {})
            self.rule_actions = d.get("rule", 0)
            self.llm_actions = d.get("llm", 0)
