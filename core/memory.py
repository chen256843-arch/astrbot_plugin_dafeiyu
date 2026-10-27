"""记忆子系统：编码 → 存储 → 巩固 → 检索 → 遗忘。

存储为有向加权图 G = (V, E, w)，每个节点附语义向量，构成图通道（扩散激活）与
向量通道（模式补全）双索引。记忆按「脑区标签」region 显式分区。
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import gua
from .util import DAY, HOUR, bigrams, char_set, clamp, containment, cosine, jaccard, now, tokens

try:  # numpy 只用于加速余弦，缺失时退化为纯 Python
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

REGIONS = {
    "flash": "闪光灯",
    "episodic": "情节",
    "semantic": "语义",
    "procedural": "程序",
    "social": "社会",
    "spatial": "空间",
    "self": "自我",
}
TAU_DAYS = {"core": 3650.0, "important": 30.0, "normal": 7.0}
W_BASE = {"owner": 4.0, "other": 1.5, "self": 1.5}
W0_CAP = 12.0
HOT_CAPACITY = 5000

# 检索融合系数
COS_W, LEX_W, PPR_W = 0.65, 0.35, 0.20
FLASH_BOOST, INTENT_BOOST = 1.45, 1.35
SPEAKER_BOOST = 1.25
GUA_DECAY = (1.0, 0.86, 0.72, 0.60)
JITTER = 0.175

# 去重阈值
DEDUPE_JACCARD = 0.60
DEDUPE_CONTAIN = 0.78

_REGION_KEYS: list[tuple[str, tuple[str, ...]]] = [
    ("procedural", ("怎么做", "如何", "步骤", "方法", "教程", "做法", "先把", "然后再")),
    ("spatial", ("在哪", "哪里", "位置", "房间", "菜棚", "码头", "水井", "老树", "崖边", "阳台", "屋子", "卧室", "厨房", "书房")),
    ("self", ("你是", "你叫", "你的名字", "你自己", "我是你", "你的性格")),
    ("social", ("朋友", "群里", "关系", "同学", "同事", "家人", "妈妈", "爸爸", "认识")),
    ("episodic", ("今天", "昨天", "前天", "那天", "上次", "刚才", "刚刚", "去了", "发生", "这周", "周末")),
    ("semantic", ("是指", "定义", "原理", "知识", "规律", "意思是", "等于")),
]
_INTENT_KEYS: list[tuple[str, tuple[str, ...]]] = [
    ("procedural", ("怎么", "如何", "步骤", "方法")),
    ("spatial", ("在哪", "哪里", "哪儿", "位置")),
    ("self", ("你是谁", "你叫", "你自己", "你的")),
    ("social", ("谁", "朋友", "关系")),
    ("episodic", ("什么时候", "那天", "上次", "记得", "以前", "昨天", "之前")),
    ("semantic", ("是什么", "为什么", "什么意思", "原理")),
]
_POS = ("开心", "高兴", "喜欢", "爱", "棒", "好吃", "谢谢", "可爱", "温暖", "舒服", "哈哈", "太好了", "幸福", "感动", "期待", "想你")
_NEG = ("难过", "伤心", "讨厌", "生气", "烦", "痛", "累", "害怕", "失望", "哭", "糟糕", "孤独", "去世", "生病", "分手", "崩溃")
_HIGH = ("爱", "生气", "害怕", "哭", "激动", "惊", "永远", "第一次", "吓", "救命", "气死", "哈哈哈", "崩溃", "去世", "分手", "求婚")
_DRIVE_KEYS = {
    "social": ("聊天", "朋友", "陪", "一起", "想你"),
    "curiosity": ("为什么", "好奇", "发现", "新的", "原来"),
    "play": ("玩", "游戏", "跑", "逛", "好玩"),
    "rest": ("睡", "累", "休息", "困", "躺"),
    "recognition": ("夸", "真棒", "厉害", "谢谢", "表扬"),
    "achievement": ("完成", "学会", "做到", "赚", "考过"),
}
_TOPIC_WORDS = (
    "猫", "鸟", "菜", "花", "书", "雨", "雪", "海", "饭", "咖啡", "茶", "工作", "考试", "电影", "音乐", "游戏",
    "生日", "旅行", "天气", "钱", "家具", "番茄", "青菜", "萝卜", "草莓",
    "菜棚", "码头", "水井", "老树", "崖边", "阳台", "屋子",
)
_TRIVIAL = {"哈哈", "嗯", "嗯嗯", "好的", "好", "早", "晚安", "在吗", "ok", "哦", "啊", "收到", "1", "？", "?"}


def affect(text: str) -> tuple[float, float]:
    """词典法近似情绪效价 / 唤醒度。"""
    p = sum(text.count(w) for w in _POS)
    n = sum(text.count(w) for w in _NEG)
    hi = sum(text.count(w) for w in _HIGH)
    excl = text.count("!") + text.count("！")
    valence = clamp((p - n) * 0.4, -1.0, 1.0)
    arousal = clamp(0.15 + 0.2 * hi + 0.12 * min(excl, 3) + 0.08 * (p + n), 0.0, 1.0)
    return valence, arousal


def classify_region(text: str) -> str:
    best, score = "episodic", 0
    for region, keys in _REGION_KEYS:
        s = sum(1 for k in keys if k in text)
        if s > score:
            best, score = region, s
    return best


def query_intent(query: str) -> str | None:
    for region, keys in _INTENT_KEYS:
        if any(k in query for k in keys):
            return region
    return None


def infer_drive(text: str) -> str | None:
    best, score = None, 0
    for d, keys in _DRIVE_KEYS.items():
        s = sum(1 for k in keys if k in text)
        if s > score:
            best, score = d, s
    return best


def extract_tags(text: str) -> list[str]:
    return [w for w in _TOPIC_WORDS if w in text]


def speaker_of(item: dict) -> str:
    """说话人：显式 speaker 字段 → user: 标签 → 角色自己（"self"）→ 未知（""）。"""
    if item.get("speaker"):
        return str(item["speaker"])
    for tag in item.get("tags", ()):
        if tag.startswith("user:"):
            return tag[5:]
    return "self" if item.get("source") == "self" else ""


@dataclass
class Hit:
    node: dict
    score: float
    parts: dict = field(default_factory=dict)
    gua: str = ""


class MemoryStore:
    def __init__(self, capacity: int = HOT_CAPACITY, rng: random.Random | None = None):
        self.capacity = capacity
        self.rng = rng or random.Random()
        self.nodes: dict[str, dict] = {}
        self.cold: dict[str, dict] = {}
        self.edges: dict[str, dict[str, float]] = defaultdict(dict)
        self.vectors: dict[str, dict] = {}  # id -> {"b": backend, "v": [...]}
        self.pool: list[dict] = []  # 候选池
        self.counter = 0
        self.last_written: str | None = None
        # 运行时缓存（不持久化）
        self._chars: dict[str, set] = {}
        self._toks: dict[str, set] = {}
        self._bis: dict[str, set] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._vec_cache: tuple | None = None

    # ------------------------------------------------------------------ 基础
    def weight(self, node: dict, t: float | None = None) -> float:
        """w(t) = w0 · exp(−Δt/τ)，Δt 自上次访问起算。"""
        t = t or now()
        tau = TAU_DAYS.get(node.get("tier", "normal"), 7.0) * DAY
        return node["w0"] * math.exp(-max(0.0, t - node["last_access"]) / tau)

    def degree(self, nid: str) -> int:
        return len(self.edges.get(nid, {}))

    def gua_of(self, nid: str) -> str:
        return gua.node_gua(self.nodes[nid], self.degree(nid))

    def _index(self, nid: str) -> None:
        text = self.nodes[nid]["text"]
        self._chars[nid] = char_set(text)
        self._bis[nid] = bigrams(text)
        toks = set(tokens(text))
        self._toks[nid] = toks
        for t in toks:
            self._df[t] += 1

    def _unindex(self, nid: str) -> None:
        for t in self._toks.pop(nid, ()):
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]
        self._chars.pop(nid, None)
        self._bis.pop(nid, None)

    def _idf(self, t: str) -> float:
        n = max(1, len(self.nodes))
        return math.log(1 + n / (1 + self._df.get(t, 0)))

    def _link(self, a: str, b: str, w: float, add: bool = False) -> None:
        if a == b:
            return
        cur = self.edges[a].get(b, 0.0)
        self.edges[a][b] = min(1.0, cur + w) if add else max(cur, min(1.0, w))

    def _remove_node(self, nid: str, to_cold: bool = False) -> None:
        node = self.nodes.pop(nid, None)
        if node is None:
            return
        self._unindex(nid)
        self.edges.pop(nid, None)
        for src in list(self.edges):
            self.edges[src].pop(nid, None)
        self.vectors.pop(nid, None)
        self._vec_cache = None
        if to_cold:
            node["cold_at"] = now()
            self.cold[nid] = node
        if self.last_written == nid:
            self.last_written = None

    # ------------------------------------------------------------------ 编码
    def surprise(self, text: str, speaker: str | None = None) -> tuple[float, str | None]:
        """s = 1 − max_i Jaccard(text, m_i)，返回 (s, 最相似节点)。给出 speaker 时只和同一人的记忆比。"""
        cs = char_set(text)
        best, best_id = 0.0, None
        for nid, other in self._chars.items():
            if speaker is not None and self.nodes[nid].get("speaker", "") != speaker:
                continue
            j = jaccard(cs, other)
            if j > best:
                best, best_id = j, nid
        return 1.0 - best, best_id

    def propose(self, text: str, source: str = "other", **meta) -> None:
        """候选池：先暂存，由 judge() 统一判断是否写入。meta 为 write() 的关键字参数。"""
        text = text.strip()
        if text:
            self.pool.append({"text": text, "source": source, "ts": now(), **meta})

    def judge(self, min_surprise: float = 0.12) -> list[dict]:
        """候选池 → 判断阶段。过短、寒暄，或同一个人几乎重复说过的（只强化旧节点）不写入。

        别人说过相似的话不算重复：谁说的本身就是要记住的信息。
        """
        written: list[dict] = []
        pool, self.pool = self.pool, []
        for c in pool:
            text = c.pop("text")
            if len(char_set(text)) < 4 or text.strip().lower() in _TRIVIAL:
                continue
            speaker = speaker_of(c)
            s, sim_id = self.surprise(text, speaker)
            if s < min_surprise and sim_id:
                self._rehearse(sim_id)
                continue
            written.append(self.write(text, **{k: v for k, v in c.items() if k != "ts"}, ts=c.get("ts")))
        return written

    def _rehearse(self, nid: str, bonus: float = 0.3) -> None:
        node = self.nodes[nid]
        node["w0"] = min(W0_CAP, self.weight(node) + bonus)
        node["last_access"] = now()

    def write(
        self,
        text: str,
        source: str = "other",
        region: str | None = None,
        tier: str | None = None,
        tags: Iterable[str] = (),
        valence: float | None = None,
        arousal: float | None = None,
        ts: float | None = None,
        link_to: Iterable[str] = (),
        speaker: str | None = None,
        speaker_name: str = "",
    ) -> dict:
        """写入一条记忆。speaker 为说话人 QQ 号；角色自己的经历为 "self"；不给时从标签推断。"""
        t = ts or now()
        s, _ = self.surprise(text)
        v, a = affect(text)
        valence = v if valence is None else valence
        arousal = a if arousal is None else arousal
        if region is None:
            region = "flash" if abs(valence) >= 0.6 and arousal >= 0.7 else classify_region(text)
        w_base = W_BASE.get(source, 1.5)
        w0 = w_base * (1 + s)
        if tier is None:
            tier = "core" if region == "flash" else "important" if w0 >= 4.0 else "normal"
        self.counter += 1
        nid = f"m{self.counter}"
        node = {
            "id": nid,
            "text": text,
            "region": region,
            "tier": tier,
            "tags": sorted(set(extract_tags(text)) | set(tags)),
            "source": source,
            "speaker": "",
            "speaker_name": speaker_name,
            "created": t,
            "last_access": t,
            "hits": 0,
            "w0": round(w0, 4),
            "surprise": round(s, 4),
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "drive": infer_drive(text),
            "recalls": [],
            "merged": 0,
        }
        node["speaker"] = speaker if speaker is not None else speaker_of(node)
        self.nodes[nid] = node
        self._index(nid)
        # 时间邻近边（有向，前向更强）
        if self.last_written and self.last_written in self.nodes:
            self._link(self.last_written, nid, 0.5)
            self._link(nid, self.last_written, 0.25)
        # 主题边：共享标签
        content_tags = {g for g in node["tags"] if ":" not in g}
        if content_tags:
            shared: list[tuple[int, str]] = []
            for oid, other in self.nodes.items():
                if oid == nid:
                    continue
                k = len(content_tags & set(other["tags"]))
                if k:
                    shared.append((k, oid))
            shared.sort(reverse=True)
            for k, oid in shared[:3]:
                self._link(nid, oid, 0.3 * k)
                self._link(oid, nid, 0.3 * k)
        for oid in link_to:
            if oid in self.nodes:
                self._link(oid, nid, 0.6)
                self._link(nid, oid, 0.6)
        self.last_written = nid
        if len(self.nodes) > self.capacity:
            self.evict_overflow()
        return node

    # ------------------------------------------------------------------ 向量
    def set_vector(self, nid: str, backend: str, vec: list[float]) -> None:
        if nid in self.nodes:
            self.vectors[nid] = {"b": backend, "v": [round(x, 6) for x in vec]}
            self._vec_cache = None

    def missing_vectors(self, backend: str) -> list[str]:
        return [nid for nid in self.nodes if self.vectors.get(nid, {}).get("b") != backend]

    def _cosines(self, qvec: list[float], backend: str) -> dict[str, float]:
        if _np is None:
            return {nid: cosine(qvec, e["v"]) for nid, e in self.vectors.items() if e["b"] == backend}
        if self._vec_cache is None or self._vec_cache[0] != backend:
            ids = [nid for nid, e in self.vectors.items() if e["b"] == backend]
            if not ids:
                return {}
            m = _np.asarray([self.vectors[i]["v"] for i in ids], dtype=_np.float32)
            norms = _np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._vec_cache = (backend, ids, m / norms)
        _, ids, m = self._vec_cache
        q = _np.asarray(qvec, dtype=_np.float32)
        qn = float(_np.linalg.norm(q)) or 1.0
        sims = m @ (q / qn)
        return dict(zip(ids, sims.tolist()))

    # ------------------------------------------------------------------ 检索
    def _lex(self, qtoks: set, qbis: set, nid: str) -> float:
        """稀有词加权重叠 + 字二元组重叠。"""
        total = sum(self._idf(t) for t in qtoks) or 1.0
        rare = sum(self._idf(t) for t in qtoks & self._toks.get(nid, set())) / total
        bi = len(qbis & self._bis.get(nid, set())) / max(1, len(qbis))
        return 0.6 * rare + 0.4 * bi

    def _ppr(self, seeds: dict[str, float], alpha: float = 0.3, iters: int = 12) -> dict[str, float]:
        """个性化 PageRank：从种子出发沿加权边扩散激活。"""
        z = sum(seeds.values()) or 1.0
        p = {k: v / z for k, v in seeds.items()}
        r = dict(p)
        for _ in range(iters):
            nxt: dict[str, float] = defaultdict(float)
            for k, v in p.items():
                nxt[k] += alpha * v
            for u, mass in r.items():
                out = self.edges.get(u)
                if not out:
                    nxt[u] += (1 - alpha) * mass
                    continue
                tot = sum(out.values()) or 1.0
                for v_, w in out.items():
                    if v_ in self.nodes:
                        nxt[v_] += (1 - alpha) * mass * w / tot
            r = nxt
        return r

    def retrieve(
        self,
        query: str,
        k: int = 6,
        query_vec: list[float] | None = None,
        backend: str | None = None,
        drive_bias: Callable[[str | None], float] | None = None,
        mood: float = 0.0,
        temperature: float = 1.0,
        update: bool = True,
        min_score: float = 0.08,
        speaker: str | None = None,
    ) -> list[Hit]:
        """speaker 为当前说话人：他自己说过的事优先想起（×SPEAKER_BOOST）。"""
        if not self.nodes:
            return []
        t = now()
        intent = query_intent(query)
        qtoks, qbis = set(tokens(query)), bigrams(query)
        cos = self._cosines(query_vec, backend) if query_vec and backend else {}

        base: dict[str, float] = {}
        parts: dict[str, dict] = {}
        for nid in self.nodes:
            lex = self._lex(qtoks, qbis, nid)
            c = max(0.0, cos.get(nid, 0.0))
            b = COS_W * c + LEX_W * lex if nid in cos else lex
            base[nid] = b
            parts[nid] = {"cos": round(c, 3), "lex": round(lex, 3)}

        seeds = dict(sorted(((n, s) for n, s in base.items() if s > 0.05), key=lambda kv: -kv[1])[:10])
        ppr = self._ppr(seeds) if seeds else {}
        pmax = max(ppr.values(), default=0.0) or 1.0

        scored: list[tuple[float, str]] = []
        for nid, node in self.nodes.items():
            pr = ppr.get(nid, 0.0) / pmax
            fused = base[nid] + PPR_W * pr
            if fused <= 0.02:
                continue
            w = self.weight(node, t)
            node_w = 0.6 + 0.4 * math.tanh(w / 3)
            d_bias = drive_bias(node.get("drive")) if drive_bias else 1.0
            v_bias = 1.0 + 0.15 * mood * node.get("valence", 0.0)
            region = node.get("region")
            r_boost = FLASH_BOOST if region == "flash" else INTENT_BOOST if intent and region == intent else 1.0
            recent = sum(1 for ts in node.get("recalls", []) if t - ts < 6 * HOUR)
            rum = 0.75 ** (recent - 1) if recent >= 2 else 1.0
            jitter = 1.0 + self.rng.uniform(-JITTER, JITTER) * temperature
            s_boost = SPEAKER_BOOST if speaker and node.get("speaker") == speaker else 1.0
            score = fused * node_w * d_bias * v_bias * r_boost * rum * s_boost * jitter
            parts[nid].update(ppr=round(pr, 3), w=round(w, 3), region=r_boost, rum=round(rum, 3), speaker=s_boost)
            scored.append((score, nid))
        scored.sort(reverse=True)
        pool = scored[: max(50, k * 8)]

        # 贪心选择：卦象多样性 + 分区注意力预算
        quota = defaultdict(lambda: max(1, math.ceil(k / 3)))
        quota["flash"] = 2
        if intent:
            quota[intent] = k
        chosen: list[Hit] = []
        gua_count: dict[str, int] = defaultdict(int)
        region_count: dict[str, int] = defaultdict(int)
        remaining = {nid: s for s, nid in pool}
        guas = {nid: self.gua_of(nid) for nid in remaining}
        while remaining and len(chosen) < k:
            best_id, best_s = None, -1.0
            for nid, s in remaining.items():
                reg = self.nodes[nid]["region"]
                if region_count[reg] >= quota[reg]:
                    continue
                g = gua_count[guas[nid]]
                adj = s * GUA_DECAY[min(g, len(GUA_DECAY) - 1)]
                if adj > best_s:
                    best_id, best_s = nid, adj
            if best_id is None or best_s < min_score:
                break
            remaining.pop(best_id)
            gua_count[guas[best_id]] += 1
            region_count[self.nodes[best_id]["region"]] += 1
            chosen.append(Hit(self.nodes[best_id], round(best_s, 4), parts[best_id], guas[best_id]))

        if update and chosen:
            self._after_recall([h.node["id"] for h in chosen], t)
        return chosen

    def _after_recall(self, ids: list[str], t: float) -> None:
        for nid in ids:
            node = self.nodes[nid]
            recalls = [ts for ts in node.get("recalls", []) if t - ts < DAY][-9:] + [t]
            node["recalls"] = recalls
            node["hits"] = node.get("hits", 0) + 1
            recent = sum(1 for ts in recalls if t - ts < 6 * HOUR)
            if recent >= 4:  # 反刍抑制：短时间反复召回 → 降权
                node["w0"] = self.weight(node, t) * 0.9
            else:  # 检索即强化
                node["w0"] = min(W0_CAP, self.weight(node, t) + 0.4)
            node["last_access"] = t
        top = ids[:4]
        for i, a in enumerate(top):  # 共激活边
            for b in top[i + 1 :]:
                self._link(a, b, 0.1, add=True)
                self._link(b, a, 0.1, add=True)

    # ------------------------------------------------------------------ 巩固
    def replay(self, t: float | None = None) -> int:
        t = t or now()
        n = 0
        for node in self.nodes.values():
            if t - node["created"] < DAY and not node.get("replayed"):
                if node.get("surprise", 0) >= 0.5 or node.get("arousal", 0) >= 0.6:
                    node["w0"] = min(W0_CAP, node["w0"] * 1.15)
                    node["replayed"] = True
                    n += 1
        return n

    def _dedupe_candidates(self) -> Iterable[tuple[str, str]]:
        inv: dict[str, list[str]] = defaultdict(list)
        for nid, bis in self._bis.items():
            for b in bis:
                inv[b].append(nid)
        seen: set[tuple[str, str]] = set()
        for nid, bis in self._bis.items():
            counts: dict[str, int] = defaultdict(int)
            for b in bis:
                lst = inv[b]
                if len(lst) > 200:
                    continue
                for o in lst:
                    if o > nid:
                        counts[o] += 1
            need = max(2, len(bis) // 3)
            for o, c in counts.items():
                if c >= need and (nid, o) not in seen:
                    seen.add((nid, o))
                    yield nid, o

    def dedupe(self) -> int:
        """Jaccard ≥ 0.60 且包含度 ≥ 0.78，且同 region 或共享标签 → 合并。闪光灯不参与。"""
        merged = 0
        for a, b in list(self._dedupe_candidates()):
            if a not in self.nodes or b not in self.nodes:
                continue
            na, nb = self.nodes[a], self.nodes[b]
            if "flash" in (na["region"], nb["region"]):
                continue
            if na.get("speaker", "") != nb.get("speaker", ""):
                continue  # 不同的人说的相似的话不合并，否则会记错是谁说的
            if na["region"] != nb["region"] and not (set(na["tags"]) & set(nb["tags"])):
                continue
            ca, cb = self._chars[a], self._chars[b]
            if jaccard(ca, cb) < DEDUPE_JACCARD or containment(ca, cb) < DEDUPE_CONTAIN:
                continue
            keep, drop = (a, b) if self.weight(na) >= self.weight(nb) else (b, a)
            self._merge(keep, drop)
            merged += 1
        return merged

    def _merge(self, keep: str, drop: str) -> None:
        k, d = self.nodes[keep], self.nodes[drop]
        if len(d["text"]) > len(k["text"]):
            self._unindex(keep)
            k["text"] = d["text"]
            self._index(keep)
            self.vectors.pop(keep, None)
        k["w0"] = min(W0_CAP, max(self.weight(k), self.weight(d)) + 0.3 * min(self.weight(k), self.weight(d)))
        k["last_access"] = now()
        k["hits"] = k.get("hits", 0) + d.get("hits", 0)
        k["tags"] = sorted(set(k["tags"]) | set(d["tags"]))
        k["merged"] = k.get("merged", 0) + 1 + d.get("merged", 0)
        for dst, w in self.edges.get(drop, {}).items():
            self._link(keep, dst, w)
        for src, out in self.edges.items():
            if drop in out:
                self._link(src, keep, out[drop])
        self._remove_node(drop)

    def abstraction_plans(self, min_size: int = 3, window_days: float = 7.0) -> list[tuple[str, list[str]]]:
        """同一个人、同一主题标签下近期 ≥3 条情节记忆 → 待抽象为一条语义记忆。

        按人分组，概括里不会把不同人的事混在一起。
        """
        t = now()
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for nid, node in self.nodes.items():
            if node["region"] in ("episodic", "flash") and t - node["created"] < window_days * DAY:
                for tag in node["tags"]:
                    if ":" not in tag and tag not in node.get("abstracted", []):
                        groups[(tag, node.get("speaker", ""))].append(nid)
        return [(tag, ids) for (tag, _), ids in groups.items() if len(ids) >= min_size]

    def apply_abstraction(self, tag: str, ids: list[str], text: str) -> dict:
        """写入概括。说话人和可见范围随来源：都来自同一个群才带群标签，否则按私聊对待。"""
        src = [self.nodes[i] for i in ids if i in self.nodes]
        speaker = src[0].get("speaker", "") if src else ""
        scope = [g for g in src[0]["tags"] if g.startswith("group:")] if src else []
        scope = [g for g in scope if all(g in n["tags"] for n in src)]
        user_tags = [f"user:{speaker}"] if speaker and speaker != "self" else []
        node = self.write(
            text,
            source="self",
            region="semantic",
            tier="important",
            tags=[tag, *user_tags, *scope],
            link_to=ids,
            speaker=speaker,
            speaker_name=src[0].get("speaker_name", "") if src else "",
        )
        for n in src:
            n.setdefault("abstracted", []).append(tag)
        return node

    def template_abstraction(self, tag: str, ids: list[str]) -> str:
        src = [self.nodes[i] for i in ids if i in self.nodes]
        snippets = [n["text"][:24] for n in src[:3]]
        who = (src[0].get("speaker_name") or "") if src else ""
        head = f"关于「{tag}」，{who}最近提到 {len(ids)} 件事：" if who else f"关于「{tag}」最近有 {len(ids)} 件事："
        return head + "；".join(snippets)

    def prune_edges(self, decay: float = 0.95, floor: float = 0.05) -> int:
        removed = 0
        for src in list(self.edges):
            out = self.edges[src]
            for dst in list(out):
                out[dst] *= decay
                if out[dst] < floor:
                    del out[dst]
                    removed += 1
            if not out:
                del self.edges[src]
        return removed

    # ------------------------------------------------------------------ 遗忘
    def evict_overflow(self) -> int:
        over = len(self.nodes) - self.capacity
        if over <= 0:
            return 0
        t = now()
        cands = sorted(
            (self.weight(n, t), nid) for nid, n in self.nodes.items() if n["region"] != "flash"
        )
        for _, nid in cands[:over]:
            self._remove_node(nid, to_cold=True)
        return min(over, len(cands))

    def forget(self, min_weight: float = 0.05, min_age_days: float = 30) -> int:
        """低权重、零命中、长龄节点清出热库（下沉冷库）。闪光灯不淘汰。"""
        t = now()
        victims = [
            nid
            for nid, n in self.nodes.items()
            if n["region"] != "flash"
            and self.weight(n, t) < min_weight
            and n.get("hits", 0) == 0
            and t - n["created"] > min_age_days * DAY
        ]
        for nid in victims:
            self._remove_node(nid, to_cold=True)
        return len(victims) + self.evict_overflow()

    def consolidate(self, summarize: Callable[[str, list[str]], str] | None = None) -> dict:
        """夜间巩固：重放 → 合并 → 抽象 → 修剪 → 迁移。"""
        report = {"replay": self.replay(), "merge": self.dedupe(), "abstract": 0}
        for tag, ids in self.abstraction_plans():
            text = summarize(tag, ids) if summarize else self.template_abstraction(tag, ids)
            self.apply_abstraction(tag, ids, text)
            report["abstract"] += 1
        report["prune"] = self.prune_edges()
        report["forget"] = self.forget()
        return report

    # ------------------------------------------------------------------ 维护
    def maintain(self) -> dict:
        """修复自环边、悬空边、缺失 region/tags、多余向量，保证图结构不变量。"""
        fixed = defaultdict(int)
        for src in list(self.edges):
            if src not in self.nodes:
                del self.edges[src]
                fixed["dangling"] += 1
                continue
            out = self.edges[src]
            if src in out:
                del out[src]
                fixed["self_loop"] += 1
            for dst in list(out):
                if dst not in self.nodes:
                    del out[dst]
                    fixed["dangling"] += 1
            if not out:
                del self.edges[src]
        for node in self.nodes.values():
            if node.get("region") not in REGIONS:
                node["region"] = classify_region(node["text"])
                fixed["region"] += 1
            if not isinstance(node.get("tags"), list):
                node["tags"] = extract_tags(node["text"])
                fixed["tags"] += 1
            if "speaker" not in node:
                node["speaker"] = speaker_of(node)
                node.setdefault("speaker_name", "")
                fixed["speaker"] += 1
        for nid in list(self.vectors):
            if nid not in self.nodes:
                del self.vectors[nid]
                fixed["extra_vector"] += 1
        if fixed.get("extra_vector"):
            self._vec_cache = None
        return dict(fixed)

    def stats(self) -> dict:
        regions: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            regions[n["region"]] += 1
        return {
            "hot": len(self.nodes),
            "cold": len(self.cold),
            "edges": sum(len(v) for v in self.edges.values()),
            "vectors": len(self.vectors),
            "pool": len(self.pool),
            "regions": dict(regions),
        }

    # ------------------------------------------------------------------ 持久化
    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "cold": self.cold,
            "edges": {k: v for k, v in self.edges.items() if v},
            "pool": self.pool,
            "counter": self.counter,
            "last_written": self.last_written,
        }

    def vectors_dict(self) -> dict:
        return self.vectors

    @classmethod
    def from_dict(cls, d: dict | None, vectors: dict | None = None, **kw) -> "MemoryStore":
        m = cls(**kw)
        if d:
            m.nodes = d.get("nodes", {})
            m.cold = d.get("cold", {})
            m.edges = defaultdict(dict, d.get("edges", {}))
            m.pool = d.get("pool", [])
            m.counter = d.get("counter", len(m.nodes))
            m.last_written = d.get("last_written")
        m.vectors = vectors or {}
        for nid, node in m.nodes.items():
            if "speaker" not in node:  # 0.4.0 之前的记忆：从标签补出说话人
                node["speaker"] = speaker_of(node)
                node.setdefault("speaker_name", "")
            m._index(nid)
        return m
