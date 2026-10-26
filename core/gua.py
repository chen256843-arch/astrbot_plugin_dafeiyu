"""卦象状态编码（六爻编码）。

六十四卦在数学上是 2^6 的离散状态空间，这里只借用其结构作为记忆的 6 位特征编码，
用于结构化索引与检索多样性控制，不涉及任何占断语义。

c(node) = (b1..b6)，b1 为初爻（最下）。下卦 = tri(b1 b2 b3)，上卦 = tri(b4 b5 b6)。
"""
from __future__ import annotations

from typing import Sequence

# 三爻（自下而上）→ 卦名
TRIGRAMS: dict[tuple[int, int, int], str] = {
    (1, 1, 1): "乾",
    (1, 1, 0): "兑",
    (1, 0, 1): "离",
    (1, 0, 0): "震",
    (0, 1, 1): "巽",
    (0, 1, 0): "坎",
    (0, 0, 1): "艮",
    (0, 0, 0): "坤",
}

_ORDER = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
# NAMES[上卦][下卦]，按 _ORDER 排列
_TABLE = {
    "乾": ["乾为天", "天泽履", "天火同人", "天雷无妄", "天风姤", "天水讼", "天山遁", "天地否"],
    "兑": ["泽天夬", "兑为泽", "泽火革", "泽雷随", "泽风大过", "泽水困", "泽山咸", "泽地萃"],
    "离": ["火天大有", "火泽睽", "离为火", "火雷噬嗑", "火风鼎", "火水未济", "火山旅", "火地晋"],
    "震": ["雷天大壮", "雷泽归妹", "雷火丰", "震为雷", "雷风恒", "雷水解", "雷山小过", "雷地豫"],
    "巽": ["风天小畜", "风泽中孚", "风火家人", "风雷益", "巽为风", "风水涣", "风山渐", "风地观"],
    "坎": ["水天需", "水泽节", "水火既济", "水雷屯", "水风井", "坎为水", "水山蹇", "水地比"],
    "艮": ["山天大畜", "山泽损", "山火贲", "山雷颐", "山风蛊", "山水蒙", "艮为山", "山地剥"],
    "坤": ["地天泰", "地泽临", "地火明夷", "地雷复", "地风升", "地水师", "地山谦", "坤为地"],
}
NAMES: dict[str, dict[str, str]] = {
    up: {low: _TABLE[up][i] for i, low in enumerate(_ORDER)} for up in _ORDER
}

# 每一爻对应的记忆特征（仅作索引标注）
BIT_MEANING = ("主人来源", "正效价", "高唤醒", "高等级", "情节性", "高连通")


def tri(bits: Sequence[int]) -> str:
    return TRIGRAMS[(int(bits[0]), int(bits[1]), int(bits[2]))]


def encode_bits(bits: Sequence[int]) -> str:
    if len(bits) != 6:
        raise ValueError("需要 6 位")
    lower = tri(bits[0:3])
    upper = tri(bits[3:6])
    return NAMES[upper][lower]


def node_bits(node: dict, degree: int) -> tuple[int, ...]:
    """把记忆节点映射为 6 位特征。"""
    return (
        int(node.get("source") == "owner"),
        int(node.get("valence", 0.0) >= 0),
        int(node.get("arousal", 0.0) >= 0.5),
        int(node.get("tier") in ("core", "important")),
        int(node.get("region") in ("episodic", "flash")),
        int(degree >= 3),
    )


def node_gua(node: dict, degree: int) -> str:
    return encode_bits(node_bits(node, degree))


def bits_str(bits: Sequence[int]) -> str:
    """自下而上的位串，例如 111111。"""
    return "".join(str(int(b)) for b in bits)
