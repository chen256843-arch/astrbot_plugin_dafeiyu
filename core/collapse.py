"""量子式决策（quantum-like decision, Busemeyer & Bruza 2012）。

纯数学借用，不涉及任何物理量子过程：
- 每个选项 k 有两条「路径」：内在倾向振幅 a_k = sqrt(pref_k)，情境振幅 b_k = sqrt(ctx_k)
- 情境相位 φ_k 决定两条路径的干涉：ψ_k = a_k + b_k · e^{iφ_k}
- 玻恩规则：P(k) = |ψ_k|² / Σ_j |ψ_j|²，然后抽样
- 确定性优先：首选 P ≥ 0.55 且领先次选 ≥ 0.15 时直接采纳，不抽样
"""
from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import Mapping

DET_THRESHOLD = 0.55
DET_MARGIN = 0.15


@dataclass
class Collapse:
    choice: str
    probs: dict[str, float]
    mode: str  # deterministic | sampled


def amplitudes(
    prefs: Mapping[str, float],
    context: Mapping[str, float] | None = None,
    phase: Mapping[str, float] | None = None,
) -> dict[str, complex]:
    context = context or {}
    phase = phase or {}
    out: dict[str, complex] = {}
    for k, p in prefs.items():
        a = math.sqrt(max(0.0, p))
        b = math.sqrt(max(0.0, context.get(k, 0.0)))
        out[k] = a + b * cmath.exp(1j * phase.get(k, 0.0))
    return out


def born(psi: Mapping[str, complex]) -> dict[str, float]:
    mags = {k: abs(v) ** 2 for k, v in psi.items()}
    total = sum(mags.values())
    if total <= 0:
        n = len(mags) or 1
        return {k: 1.0 / n for k in mags}
    return {k: v / total for k, v in mags.items()}


def collapse(
    prefs: Mapping[str, float],
    rng: random.Random | None = None,
    context: Mapping[str, float] | None = None,
    phase: Mapping[str, float] | None = None,
    deterministic_first: bool = True,
    threshold: float = DET_THRESHOLD,
    margin: float = DET_MARGIN,
) -> Collapse:
    if not prefs:
        raise ValueError("选项为空")
    rng = rng or random
    probs = born(amplitudes(prefs, context, phase))
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if deterministic_first and p1 >= threshold and p1 - p2 >= margin:
        return Collapse(top, probs, "deterministic")
    r = rng.random()
    acc = 0.0
    for k, p in ranked:
        acc += p
        if r <= acc:
            return Collapse(k, probs, "sampled")
    return Collapse(ranked[-1][0], probs, "sampled")


def pick(options: Mapping[str, float], rng: random.Random | None = None) -> str:
    """无情境的便捷抽样（仍走确定性优先）。"""
    return collapse(options, rng).choice
