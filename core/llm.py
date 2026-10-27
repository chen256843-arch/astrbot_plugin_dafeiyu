"""对话接口 + 成本记账（日耗、剩余可用天数、缓存命中率）。

两种后端：直接调用 OpenAI 兼容接口（DeepSeek 等），或由宿主框架注入的 backend
（例如 AstrBot 里已经配好的提供商）。backend 签名：
    async (messages, max_tokens, temperature) -> (text | None, 输入 token, 输出 token[, 其中命中缓存的输入 token])

计费：输入里命中前缀缓存的部分按「命中单价」，其余按「未命中单价」，输出按「输出单价」：
    费用 = 命中 × price_in_hit + (输入 − 命中) × price_in + 输出 × price_out   （单价均为 元 / 千 token）
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

log = logging.getLogger("dafeiyu.llm")

Backend = Callable[[list[dict], int, float], Awaitable[tuple]]


def today() -> str:
    return time.strftime("%Y-%m-%d")


def usage_from_openai(usage: dict | None) -> tuple[int, int, int]:
    """从 OpenAI 兼容接口的 usage 里取 (输入, 输出, 命中缓存的输入)。

    DeepSeek 用 prompt_cache_hit_tokens；OpenAI 风格用 prompt_tokens_details.cached_tokens。
    """
    usage = usage or {}
    tin = int(usage.get("prompt_tokens") or 0)
    tout = int(usage.get("completion_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return tin, tout, min(int(hit or 0), tin)


class LLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        price_in: float = 0.002,
        price_out: float = 0.008,
        backend: Backend | None = None,
        price_in_hit: float = 0.0005,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.price_in = price_in  # 输入未命中缓存，每千 token 价格（元）
        self.price_in_hit = price_in_hit  # 输入命中缓存
        self.price_out = price_out
        self.backend = backend
        self.usage: dict[str, dict] = {}
        self.balance: dict = {}  # {"value": 余额, "ts": 记录时间, "day": 日期}
        self.rule_actions = 0
        self.llm_actions = 0

    @property
    def enabled(self) -> bool:
        return bool(self.backend or self.api_key)

    def cost_of(self, tin: int, tout: int, hit: int = 0) -> float:
        hit = min(hit, tin)
        return (tin - hit) / 1000 * self.price_in + hit / 1000 * self.price_in_hit + tout / 1000 * self.price_out

    def _record(self, tin: int, tout: int, hit: int = 0) -> None:
        u = self.usage.setdefault(today(), {"calls": 0, "in": 0, "out": 0, "hit": 0, "cost": 0.0})
        u["calls"] += 1
        u["in"] += tin
        u["out"] += tout
        u["hit"] = u.get("hit", 0) + min(hit, tin)
        u["cost"] = round(u["cost"] + self.cost_of(tin, tout, hit), 6)
        if len(self.usage) > 60:
            for k in sorted(self.usage)[:-60]:
                del self.usage[k]

    async def chat(
        self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.8, backend: Backend | None = None
    ) -> str | None:
        """backend 可临时指定（例如整理卡片用另一个模型），用量照样记在这里。"""
        backend = backend or self.backend
        if backend:
            try:
                res = await backend(messages, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM 调用失败: %s", e)
                return None
            text, tin, tout = res[0], res[1], res[2]
            self._record(tin, tout, res[3] if len(res) > 3 else 0)
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
        tin, tout, hit = usage_from_openai(data.get("usage"))
        self._record(tin, tout, hit)
        return (data["choices"][0]["message"].get("content") or "").strip() or None

    def daily_burn(self, days: int = 7) -> float:
        recent = [self.usage[k]["cost"] for k in sorted(self.usage)[-days:]]
        return sum(recent) / len(recent) if recent else 0.0

    def report(self) -> str:
        u = self.usage.get(today(), {"calls": 0, "in": 0, "out": 0, "hit": 0, "cost": 0.0})
        hit = u.get("hit", 0)
        rate = hit / u["in"] * 100 if u["in"] else 0.0
        saved = hit / 1000 * (self.price_in - self.price_in_hit)
        recent = [self.usage[k] for k in sorted(self.usage)[-7:]]
        week_in = sum(x["in"] for x in recent)
        week_rate = sum(x.get("hit", 0) for x in recent) / week_in * 100 if week_in else 0.0
        burn = self.daily_burn()
        total = self.rule_actions + self.llm_actions
        ratio = self.rule_actions / total * 100 if total else 0.0
        lines = [
            f"今日调用 {u['calls']} 次，输入 {u['in']}（命中缓存 {hit}）/ 输出 {u['out']} token，约 ¥{u['cost']:.4f}",
            f"缓存命中率：今日 {rate:.0f}%，近 7 日 {week_rate:.0f}%；今日因命中省下约 ¥{saved:.4f}",
            f"单价（元/千 token）：输入未命中 {self.price_in}，命中 {self.price_in_hit}，输出 {self.price_out}",
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
