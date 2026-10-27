"""气象：Open-Meteo（免密钥）按 8 个地点每小时更新；请求失败时用坍缩抽样模拟。"""
from __future__ import annotations

import logging
import random

from .collapse import pick
from .util import now

log = logging.getLogger("dafeiyu.weather")

API = "https://api.open-meteo.com/v1/forecast"
FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "uv_index",
    "surface_pressure",
)
# 地点 → (纬度, 经度)
LOCATIONS: dict[str, tuple[float, float]] = {
    "海": (18.25, 109.50),
    "云海": (30.13, 118.17),
    "森林": (21.95, 100.80),
    "雪山": (27.10, 100.20),
    "城市": (31.23, 121.47),
    "沙漠": (40.14, 94.66),
    "极夜": (69.65, 18.96),
    "旧城": (37.20, 112.18),
}


def sky(w: dict) -> str:
    return "下雨" if w.get("precipitation", 0) > 0.2 else "阴" if w.get("cloud", 0) > 0.7 else "多云" if w.get("cloud", 0) > 0.3 else "晴"


def describe(w: dict) -> str:
    if not w:
        return "天气未知"
    return f"{sky(w)} {w.get('temp', 0):.0f}°C 湿度{w.get('humidity', 0):.0f}% 风{w.get('wind', 0):.0f}km/h"


def describe_short(w: dict) -> str:
    """给模型看的：只要天色和温度。"""
    return f"{sky(w)} {w.get('temp', 0):.0f}°C" if w else ""


async def fetch(lat: float, lon: float) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(API, params={"latitude": lat, "longitude": lon, "current": ",".join(FIELDS)})
        r.raise_for_status()
        c = r.json()["current"]
    return {
        "temp": c["temperature_2m"],
        "humidity": c["relative_humidity_2m"],
        "wind": c["wind_speed_10m"],
        "cloud": c["cloud_cover"] / 100,
        "precipitation": c["precipitation"],
        "uv": c.get("uv_index", 0) or 0,
        "pressure": c.get("surface_pressure", 1013),
        "ts": now(),
        "simulated": False,
    }


def simulate(prev: dict | None, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    prev = prev or {"temp": 22, "humidity": 70, "wind": 10, "cloud": 0.4, "precipitation": 0, "uv": 3, "pressure": 1012}
    sky = pick({"晴": 0.4, "多云": 0.3, "阴": 0.15, "雨": 0.15}, rng)
    cloud = {"晴": 0.1, "多云": 0.45, "阴": 0.85, "雨": 0.95}[sky]
    return {
        "temp": prev["temp"] + rng.uniform(-1.5, 1.5),
        "humidity": max(20, min(100, prev["humidity"] + rng.uniform(-5, 5) + (15 if sky == "雨" else 0))),
        "wind": max(0, prev["wind"] + rng.uniform(-3, 3)),
        "cloud": cloud,
        "precipitation": rng.uniform(0.5, 4) if sky == "雨" else 0.0,
        "uv": max(0, (1 - cloud) * 8),
        "pressure": prev.get("pressure", 1012) + rng.uniform(-1, 1),
        "ts": now(),
        "simulated": True,
    }


async def update_all(current: dict[str, dict], rng: random.Random | None = None) -> dict[str, dict]:
    out = dict(current)
    for name, (lat, lon) in LOCATIONS.items():
        try:
            out[name] = await fetch(lat, lon)
        except Exception as e:  # noqa: BLE001
            log.warning("Open-Meteo %s 获取失败，改用模拟: %s", name, e)
            out[name] = simulate(current.get(name), rng)
    return out
