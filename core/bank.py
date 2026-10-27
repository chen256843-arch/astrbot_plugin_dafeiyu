"""大记忆：每个人一个小记忆库，再加上她自己的经历库。

- 小记忆库就是原来的 MemoryStore，编码、遗忘曲线、巩固、去重都不变，只是各管各的人；
  所以一条记忆不必再标「是谁说的」，它在谁的库里就是谁的。
- 大记忆负责路由：根据问了什么、谁在问，决定调用哪几个小库，必要时联合调用后合并排序。
- 热库容量是全局的：所有小库加起来超过上限时，在全体里按当前权重淘汰最弱的。
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .memory import HOT_CAPACITY, Hit, MemoryStore, speaker_of
from .util import atomic_write_json, now, read_json

SELF = "self"  # 她自己的经历
FANOUT_WORDS = ("大家", "谁", "群里", "他们", "她们", "别人", "其他人", "群友")
MAX_FANOUT = 8
W_SPEAKER, W_MENTIONED, W_SELF, W_FANOUT = 1.0, 1.0, 0.9, 0.8
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class BankHit:
    key: str  # 小库：说话人 QQ 号，或 SELF
    hit: Hit
    score: float

    @property
    def node(self) -> dict:
        return self.hit.node


def _filename(key: str) -> str:
    if _SAFE.match(key):
        return key
    return "k" + hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


class MemoryBank:
    def __init__(self, capacity: int = HOT_CAPACITY, rng: random.Random | None = None):
        self.capacity = capacity
        self.rng = rng or random.Random()
        self.stores: dict[str, MemoryStore] = {}
        self.meta: dict[str, dict] = {}  # key -> {"name": 称呼, "last": 最后写入时间}

    # ------------------------------------------------------------------ 小库
    def store(self, key: str, name: str = "") -> MemoryStore:
        """取（必要时新建）某人的小库。单库不设容量，容量由大记忆统一管。"""
        key = str(key) or SELF
        if key not in self.stores:
            self.stores[key] = MemoryStore(capacity=10**9, rng=self.rng)
            self.meta.setdefault(key, {"name": "", "last": 0.0})
        if name:
            self.meta[key]["name"] = name
        return self.stores[key]

    def name(self, key: str) -> str:
        return self.meta.get(key, {}).get("name", "")

    def propose(self, key: str, text: str, name: str = "", **kw) -> None:
        kw.setdefault("speaker", key)
        kw.setdefault("speaker_name", name)
        self.store(key, name).propose(text, **kw)

    def write(self, key: str, text: str, name: str = "", **kw) -> dict:
        kw.setdefault("speaker", key)
        kw.setdefault("speaker_name", name)
        node = self.store(key, name).write(text, **kw)
        self.meta[key]["last"] = now()
        self.evict_overflow()
        return node

    # ------------------------------------------------------------------ 定时任务
    def judge(self) -> list[tuple[str, dict]]:
        written = []
        for key, st in self.stores.items():
            for node in st.judge():
                written.append((key, node))
                self.meta[key]["last"] = now()
        self.evict_overflow()
        return written

    def total(self) -> int:
        return sum(len(st.nodes) for st in self.stores.values())

    def evict_overflow(self) -> int:
        """全体热库超过上限时，跨库淘汰当前权重最低的非闪光灯节点（下沉到各自冷库）。"""
        over = self.total() - self.capacity
        if over <= 0:
            return 0
        t = now()
        cands = sorted(
            (st.weight(n, t), key, nid)
            for key, st in self.stores.items()
            for nid, n in st.nodes.items()
            if n["region"] != "flash"
        )
        for _, key, nid in cands[:over]:
            self.stores[key]._remove_node(nid, to_cold=True)
        return min(over, len(cands))

    def missing_vectors(self, backend: str) -> list[tuple[str, str]]:
        return [(key, nid) for key, st in self.stores.items() for nid in st.missing_vectors(backend)]

    def maintain(self) -> dict:
        total: dict[str, int] = {}
        for st in self.stores.values():
            for k, v in st.maintain().items():
                total[k] = total.get(k, 0) + v
        return total

    def abstraction_plans(self) -> list[tuple[str, str, list[str]]]:
        return [(key, tag, ids) for key, st in self.stores.items() for tag, ids in st.abstraction_plans()]

    def consolidate(self, summarize: Callable[[str, str, list[str]], str | None] | None = None) -> dict:
        """各小库分别巩固（重放、合并、抽象、修剪、迁移），再统一做容量淘汰。"""
        report: dict[str, int] = {}
        for key, st in self.stores.items():

            def summ(tag: str, ids: list[str], key=key, st=st) -> str:
                return (summarize(key, tag, ids) if summarize else None) or st.template_abstraction(tag, ids)

            for k, v in st.consolidate(summ).items():
                report[k] = report.get(k, 0) + v
        report["forget"] = report.get("forget", 0) + self.evict_overflow()
        return report

    def stats(self) -> dict:
        regions: dict[str, int] = {}
        out = {"hot": 0, "cold": 0, "edges": 0, "vectors": 0, "pool": 0}
        for st in self.stores.values():
            s = st.stats()
            for k in out:
                out[k] += s[k]
            for r, n in s["regions"].items():
                regions[r] = regions.get(r, 0) + n
        out["regions"] = regions
        out["people"] = sum(1 for k in self.stores if k != SELF)
        return out

    # ------------------------------------------------------------------ 路由与检索
    def route(self, query: str, speaker: str, aliases: dict[str, Iterable[str]]) -> dict[str, float]:
        """决定这次要调用哪些小库及其权重。

        - 当前说话人的库：1.0
        - 她自己的经历库：0.9
        - 问题里点了名的人（昵称或「主人」）：1.0
        - 问「大家 / 谁 / 群里…」：联合调用最近活跃的人，0.8
        """
        weights: dict[str, float] = {}
        if speaker in self.stores:
            weights[speaker] = W_SPEAKER
        if SELF in self.stores:
            weights[SELF] = W_SELF
        for key, names in aliases.items():
            if key in self.stores and key not in weights and any(n and len(n) >= 2 and n in query for n in names):
                weights[key] = W_MENTIONED
        if any(w in query for w in FANOUT_WORDS):
            recent = sorted(
                (k for k in self.stores if k not in weights and k != SELF),
                key=lambda k: self.meta.get(k, {}).get("last", 0.0),
                reverse=True,
            )
            for key in recent[:MAX_FANOUT]:
                weights[key] = W_FANOUT
        return weights

    def retrieve(
        self,
        query: str,
        weights: dict[str, float],
        k: int = 6,
        visible: Callable[[str, dict], bool] | None = None,
        update: bool = True,
        **kw,
    ) -> list[BankHit]:
        """按路由结果分别检索各小库，乘以库权重后合并取前 k，再对选中的节点做「检索即强化」。"""
        merged: list[BankHit] = []
        for key, w in weights.items():
            st = self.stores.get(key)
            if not st:
                continue
            for h in st.retrieve(query, k=k, update=False, **kw):
                if visible and not visible(key, h.node):
                    continue
                merged.append(BankHit(key, h, round(h.score * w, 4)))
        merged.sort(key=lambda b: b.score, reverse=True)
        chosen = merged[:k]
        if update and chosen:
            t = now()
            by_key: dict[str, list[str]] = {}
            for b in chosen:
                by_key.setdefault(b.key, []).append(b.node["id"])
            for key, ids in by_key.items():
                self.stores[key]._after_recall(ids, t)
        return chosen

    # ------------------------------------------------------------------ 持久化
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        index = {"version": 5, "meta": {}}
        for key, st in self.stores.items():
            fname = _filename(key)
            index["meta"][key] = {**self.meta.get(key, {}), "file": fname}
            atomic_write_json(directory / f"{fname}.json", st.to_dict())
            atomic_write_json(directory / f"{fname}.vectors.json", st.vectors_dict())
        atomic_write_json(directory / "index.json", index)

    @classmethod
    def load(cls, directory: Path, capacity: int = HOT_CAPACITY, rng: random.Random | None = None) -> "MemoryBank":
        bank = cls(capacity, rng)
        index = read_json(directory / "index.json", None)
        if not index:
            return bank
        for key, m in index.get("meta", {}).items():
            fname = m.get("file") or _filename(key)
            st = MemoryStore.from_dict(
                read_json(directory / f"{fname}.json", None),
                read_json(directory / f"{fname}.vectors.json", {}),
                capacity=10**9,
                rng=bank.rng,
            )
            bank.stores[key] = st
            bank.meta[key] = {"name": m.get("name", ""), "last": m.get("last", 0.0)}
        return bank

    @classmethod
    def from_legacy(cls, data: dict, vectors: dict, capacity: int = HOT_CAPACITY, rng: random.Random | None = None) -> "MemoryBank":
        """把 0.4 及以前的单一记忆库按说话人拆成小库。边只保留同一小库内部的。"""
        old = MemoryStore.from_dict(data, vectors)
        bank = cls(capacity, rng)

        def key_of(node: dict) -> str:
            return node.get("speaker") or SELF

        for nid, node in old.nodes.items():
            key = key_of(node)
            st = bank.store(key, node.get("speaker_name", "") if key != SELF else "")
            st.nodes[nid] = node
            st._index(nid)
            if nid in old.vectors:
                st.vectors[nid] = old.vectors[nid]
            bank.meta[key]["last"] = max(bank.meta[key]["last"], node.get("created", 0.0))
        for nid, node in old.cold.items():
            bank.store(key_of(node)).cold[nid] = node
        for c in old.pool:
            bank.store(speaker_of(c) or SELF).pool.append(c)
        for src, out in old.edges.items():
            if src not in old.nodes:
                continue
            st = bank.stores[key_of(old.nodes[src])]
            for dst, w in out.items():
                if dst in st.nodes and dst != src:
                    st.edges[src][dst] = w
        for st in bank.stores.values():
            st.counter = old.counter  # 沿用旧计数，新节点 id 不会和迁移来的重复
        return bank


def time_label(ts: float, now_dt, tz) -> str:
    """大概时间：刚才 / 今天 / 昨天 / 前天 / N 天前 / 上周 / N 周前 / N 个月前 / N 年前。"""
    from datetime import datetime

    then = datetime.fromtimestamp(ts, tz)
    secs = (now_dt - then).total_seconds()
    if secs < 3600:
        return "刚才"
    days = (now_dt.date() - then.date()).days
    if days <= 0:
        return "今天"
    if days == 1:
        return "昨天"
    if days == 2:
        return "前天"
    if days < 7:
        return f"{days}天前"
    if days < 14:
        return "上周"
    if days < 30:
        return f"{days // 7}周前"
    if days < 365:
        return f"{max(1, round(days / 30))}个月前"
    return f"{math.floor(days / 365)}年前"
