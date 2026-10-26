"""太阳几何（天文近似）。

sin h = sin φ sin δ + cos φ cos δ cos H
cos H₀ = −tan φ tan δ  → 昼长 = 2H₀ / 15 小时
I = max(0, sin h) · (1 − cloud)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone


def declination(doy: int) -> float:
    """赤纬 δ（弧度）。"""
    return math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)


def hour_angle(dt_utc: datetime, lon_deg: float) -> float:
    """时角 H（弧度），用经度换算地方平太阳时，忽略时差方程。"""
    solar_hours = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600 + lon_deg / 15
    return math.radians(15 * (solar_hours - 12))


def sin_elevation(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    phi = math.radians(lat_deg)
    delta = declination(dt_utc.timetuple().tm_yday)
    h = hour_angle(dt_utc, lon_deg)
    return math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)


def elevation_deg(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elevation(lat_deg, lon_deg, dt)))))


def day_length(lat_deg: float, doy: int) -> float:
    """昼长（小时）。极昼 24，极夜 0。"""
    x = -math.tan(math.radians(lat_deg)) * math.tan(declination(doy))
    if x <= -1:
        return 24.0
    if x >= 1:
        return 0.0
    return 2 * math.degrees(math.acos(x)) / 15


def irradiance(sin_h: float, cloud: float) -> float:
    """相对辐射强度 [0,1]，cloud ∈ [0,1]。"""
    return max(0.0, sin_h) * (1 - max(0.0, min(1.0, cloud)))
