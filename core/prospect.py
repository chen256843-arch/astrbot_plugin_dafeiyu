"""前瞻记忆（prospective memory）：认出别人说的、带日期的计划，过了那天再问问结果。

只用规则匹配，不调用模型：「下周三考驾照」「明天面试」「3月5号生日」「三天后出差」「月底搬家」。
问她的话（「你明天干嘛？」）不算。
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

EVENT_WORDS = (
    "要", "得", "去", "考", "面试", "出差", "约", "开会", "上班", "回", "旅游", "旅行", "搬家", "比赛", "答辩",
    "体检", "手术", "生日", "见", "飞", "出发", "报名", "交", "截止", "上课", "聚餐", "演出", "婚礼", "入职",
    "离职", "复查", "看病", "出院", "放假",
)
_CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_NUM = r"(\d{1,2}|[一二两三四五六七八九十]{1,3})"
MAX_AHEAD_DAYS = 120


def cn_int(s: str) -> int | None:
    """1–99 的阿拉伯或中文数字。"""
    if s.isdigit():
        return int(s)
    if not s or any(c not in _CN and c != "十" for c in s):
        return None
    if "十" not in s:
        return _CN[s] if len(s) == 1 else None
    tens, _, ones = s.partition("十")
    return (_CN.get(tens, 1) if tens else 1) * 10 + (_CN.get(ones, 0) if ones else 0)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _due(text: str, today: date) -> date | None:
    m = re.search(_NUM + r"月" + _NUM + r"[日号]", text)
    if m:
        mo, d = cn_int(m.group(1)), cn_int(m.group(2))
        if mo and d:
            due = _safe_date(today.year, mo, d)
            if due and due < today:
                due = _safe_date(today.year + 1, mo, d)
            return due
    for word, days in (("大后天", 3), ("后天", 2), ("明天", 1), ("明晚", 1), ("明早", 1)):
        if word in text:
            return today + timedelta(days=days)
    m = re.search(r"(下下|下个|下|这个|这|本)?(?:周|星期|礼拜)([一二三四五六日天])", text)
    if m:
        wd = _WEEKDAY[m.group(2)]
        monday = today - timedelta(days=today.weekday())
        prefix = m.group(1) or ""
        weeks = 2 if prefix == "下下" else 1 if prefix.startswith("下") else 0
        due = monday + timedelta(weeks=weeks, days=wd)
        if weeks == 0 and due < today and not prefix:
            due += timedelta(weeks=1)  # 「周三」说的是已经过去的这周三，那就是下周三
        return due
    m = re.search(r"(\d{1,3}|[一二两三四五六七八九十]{1,3})天(?:后|以后|之后)", text)
    if m:
        n = cn_int(m.group(1))
        return today + timedelta(days=n) if n else None
    if "月底" in text:
        return date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
    m = re.search(_NUM + r"[日号](?!线)", text)
    if m:
        d = cn_int(m.group(1))
        if d:
            due = _safe_date(today.year, today.month, d)
            if due and due < today:
                y, mo = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
                due = _safe_date(y, mo, d)
            return due
    return None


def extract(text: str, today: date) -> tuple[date, str] | None:
    """返回 (日期, 原话摘要)；不是带日期的计划则返回 None。"""
    if "你" in text and any(q in text for q in ("吗", "？", "?", "呢")):
        return None
    if not any(w in text for w in EVENT_WORDS):
        return None
    due = _due(text, today)
    if not due or due < today or (due - today).days > MAX_AHEAD_DAYS:
        return None
    return due, text.strip()[:40]


def day_label(due: date, today: date) -> str:
    """相对今天的说法：今天 / 明天 / 后天 / N天后 / 昨天 / 前天 / N天前。"""
    n = (due - today).days
    return {0: "今天", 1: "明天", 2: "后天", -1: "昨天", -2: "前天"}.get(n, f"{n}天后" if n > 0 else f"{-n}天前")
