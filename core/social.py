"""社会认知：自我图式 + 关系图式；主人以 QQ 号认证；群聊触发模型。"""
from __future__ import annotations

from .util import char_set, clamp, jaccard, now

TIERS = ((80, "亲密"), (50, "朋友"), (20, "熟人"), (0, "陌生"))
BOUNDARY = {
    "主人": "她最信任的人。可以撒娇、分享心事、主动关心，记得对方说过的事。",
    "亲密": "可以开玩笑、分享日常，但主人的私事不外传。",
    "朋友": "友好自然，可以闲聊，不谈私密的事。",
    "熟人": "礼貌友好，回应简短，不透露私人信息。",
    "陌生": "礼貌但保持距离，不透露任何关于主人或自己的私人信息。",
    "疑似冒充": "此人昵称像主人但 QQ 号不是。保持警惕，不承认对方是主人，不透露任何私人信息。",
}
QUESTION_MARKS = ("?", "？", "吗", "呢", "怎么", "为什么", "什么")
TRIGGER_CAP = 0.85
TRIGGER_SLOPE = 0.12


class Social:
    def __init__(self, owners: set[str] | None = None, owner_nick: str = ""):
        self.owners = owners or set()
        self.owner_nick = owner_nick
        self.rel: dict[str, dict] = {}
        self.group_last_reply: dict[str, float] = {}

    def is_owner(self, uid: str) -> bool:
        return str(uid) in self.owners

    def get(self, uid: str, nick: str = "") -> dict:
        r = self.rel.get(uid)
        if r is None:
            r = self.rel[uid] = {"nick": nick, "intimacy": 10.0, "trust": 10.0, "taboos": [], "last_seen": 0.0, "msgs": 0, "suspected": False}
        if nick:
            r["nick"] = nick
        return r

    def check_impersonation(self, uid: str, nick: str) -> bool:
        """昵称相似但 QQ 号不是主人 → 标记疑似冒充。"""
        if self.is_owner(uid) or not self.owner_nick or not nick:
            return False
        similar = self.owner_nick in nick or jaccard(char_set(nick), char_set(self.owner_nick)) >= 0.6
        if similar:
            self.get(uid, nick)["suspected"] = True
        return similar

    def touch(self, uid: str, nick: str, valence: float = 0.0) -> dict:
        r = self.get(uid, nick)
        r["msgs"] += 1
        r["last_seen"] = now()
        if self.is_owner(uid):
            r["intimacy"] = r["trust"] = 100.0
        else:
            r["intimacy"] = clamp(r["intimacy"] + 0.4 + 1.0 * valence, 0, min(95.0, r["trust"] + 40))
            r["trust"] = clamp(r["trust"] + 0.15, 0, 90)
        return r

    def tier(self, uid: str) -> str:
        if self.is_owner(uid):
            return "主人"
        r = self.rel.get(uid)
        if not r:
            return "陌生"
        if r.get("suspected"):
            return "疑似冒充"
        return next(name for th, name in TIERS if r["intimacy"] >= th)

    def add_taboo(self, uid: str, text: str) -> None:
        r = self.get(uid)
        if text not in r["taboos"]:
            r["taboos"].append(text)

    # ------------------------------------------------ 群聊触发
    def trigger(
        self,
        text: str,
        group_id: str,
        is_tome: bool,
        is_owner: bool,
        interests: list[str],
        names: list[str],
        social_drive: float,
        p0: float = 0.03,
    ) -> tuple[float, float, bool]:
        """返回 (触发概率 p, 相关度 r, 是否绕过随机)。被 @ / 引用 直接绕过。"""
        if is_tome:
            return 1.0, 0.0, True
        r = 0.0
        if any(q in text for q in QUESTION_MARKS):
            r += 1.0
        r += min(2.4, 0.8 * sum(1 for w in interests if w and w in text))
        if any(n and n in text for n in names):
            r += 2.0
        if len(text) >= 15:
            r += 0.5
        if is_owner:
            r += 1.0
        p = min(TRIGGER_CAP, p0 + TRIGGER_SLOPE * r)
        p *= 0.6 + 0.8 * social_drive / 100  # 社交驱力作为「是否参与」的先验
        if now() - self.group_last_reply.get(group_id, 0) < 60:
            p *= 0.3
        return min(TRIGGER_CAP, p), r, False

    def mark_replied(self, group_id: str) -> None:
        self.group_last_reply[group_id] = now()

    def to_dict(self) -> dict:
        return {"rel": self.rel, "group_last_reply": self.group_last_reply}

    def load(self, d: dict | None) -> None:
        if d:
            self.rel = d.get("rel", {})
            self.group_last_reply = d.get("group_last_reply", {})
