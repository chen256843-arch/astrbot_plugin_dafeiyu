"""通用工具：字符集相似度、分词、时间常量、原子写盘。"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

HOUR = 3600.0
DAY = 86400.0

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)
_LATIN = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[一-鿿]")


def now() -> float:
    return time.time()


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower())


def char_set(text: str) -> set[str]:
    return set(normalize(text))


def bigrams(text: str) -> set[str]:
    s = normalize(text)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def tokens(text: str) -> list[str]:
    """中文按单字 + 二元组，拉丁按整词。"""
    t = text.lower()
    out = [w for w in _LATIN.findall(t)]
    cjk = "".join(_CJK.findall(t))
    out.extend(cjk)
    out.extend(cjk[i : i + 2] for i in range(len(cjk) - 1))
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    """包含度：交集 / 较小集合。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a)
    b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
