"""印象卡：对每个人一张，对她自己一张（参考 Letta 的常驻记忆块、MemoryBank 的用户画像、MemoryOS 的长期画像）。

人记别人，靠的主要是印象（是谁、什么关系、最近怎样、喜欢什么），具体的事要被提起才想得起来。
卡片每晚由模型批量整理一次（类似 Letta 的「睡眠时」整理），白天不变，放在提示词里能命中缓存。

事实带有效期（参考 Graphiti 的双时间模型）：事实变了不删除，旧的标上 until，变成「以前…」。
"""
from __future__ import annotations

import json
import re

PROFILE_MAX = 80
FACT_MAX = 24
FACTS_CURRENT_MAX = 12
FACTS_PAST_MAX = 6
SELF_MAX = 60
INSIGHT_MAX = 30
INSIGHTS_KEEP = 3


def empty_card() -> dict:
    return {"profile": "", "facts": [], "updated": 0.0}


def current_facts(card: dict) -> list[dict]:
    return [f for f in card.get("facts", []) if not f.get("until")]


def past_facts(card: dict) -> list[dict]:
    return [f for f in card.get("facts", []) if f.get("until")]


def apply_ops(card: dict, ops: list, t: float) -> dict:
    """把模型给的修改应用到事实列表。id 是提示词里「现有事实」的序号（从 1 开始）。

    add：新事实；update：旧的标为以前、加上新的；remove：标为不再成立。
    """
    done = {"add": 0, "update": 0, "remove": 0}
    cur = current_facts(card)
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict):
            continue
        kind = op.get("op")
        text = str(op.get("text", "")).strip()[:FACT_MAX]
        idx = op.get("id")
        old = cur[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(cur) else None
        if kind == "add" and text and all(f["text"] != text for f in current_facts(card)):
            card["facts"].append({"text": text, "since": t, "until": None})
            done["add"] += 1
        elif kind == "update" and old and text and text != old["text"]:
            old["until"] = t
            card["facts"].append({"text": text, "since": t, "until": None})
            done["update"] += 1
        elif kind == "remove" and old:
            old["until"] = t
            done["remove"] += 1
    # 现有事实保留最新的若干条；以前的事实只留最近几条
    cur = current_facts(card)[-FACTS_CURRENT_MAX:]
    past = sorted(past_facts(card), key=lambda f: f["until"])[-FACTS_PAST_MAX:]
    card["facts"] = sorted(past + cur, key=lambda f: f["since"])
    return done


def render(card: dict | None, max_chars: int = 200) -> str:
    """给模型看的一行：印象｜记得的事实（以前：…）。"""
    if not card or not (card.get("profile") or current_facts(card)):
        return ""
    text = card.get("profile", "")
    facts = [f["text"] for f in current_facts(card)]
    if facts:
        text += ("｜" if text else "") + "；".join(facts)
    past = [f["text"] for f in sorted(past_facts(card), key=lambda f: f["until"])[-2:]]
    if past:
        text += "（以前：" + "；".join(past) + "）"
    insights = card.get("insights") or []
    if insights:
        text += "｜看法：" + insights[-1]["text"]
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def add_insight(card: dict, text: str, t: float) -> str:
    text = str(text or "").strip()[:INSIGHT_MAX]
    if text:
        card.setdefault("insights", []).append({"text": text, "ts": t})
        del card["insights"][:-INSIGHTS_KEEP]
    return text


def parse_json(text: str | None) -> dict | None:
    """从模型输出里取出 JSON（容忍 ```json 包裹和前后多余文字）。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def update_messages(
    bot_name: str, who: str, card: dict, memories: list[str], recap: str = "", reflect: bool = False
) -> list[dict]:
    """让模型更新对某人的印象卡。who 是这个人的称呼（主人 / 昵称）。

    reflect：这段时间重要的事攒够了，顺带要一条更高层的看法（参考 Generative Agents 的反思）。
    """
    facts = "\n".join(f"{i}. {f['text']}" for i, f in enumerate(current_facts(card), 1)) or "（无）"
    shape = '{"profile": "...", "ops": [...], "insight": "..."}' if reflect else '{"profile": "...", "ops": [...]}'
    system = (
        f"你在帮陪伴角色「{bot_name}」整理她对「{who}」的印象卡。根据现有卡片和新的记忆，只输出一个 JSON：\n"
        f"{shape}\n"
        f"- profile：{PROFILE_MAX} 字以内，第三人称，写{who}是谁、和她的关系、最近在忙什么、有什么还没了结的事。\n"
        f"- ops：对「现有事实」的修改。事实是关于{who}本人、比较稳定的信息（生日、喜好、职业、家人、住处、习惯等），"
        f"每条 {FACT_MAX} 字以内。\n"
        '  新事实 {"op":"add","text":"..."}；事实变了 {"op":"update","id":序号,"text":"新的"}；'
        '不再成立 {"op":"remove","id":序号}。没有变化就给空列表。\n'
        "- 只根据记忆里明确说过的内容，不要猜测；临时的小事不算事实。"
    )
    if reflect:
        system += (
            f"\n- insight：{INSIGHT_MAX} 字以内，她从这些事里得出的一条更高层的看法或该怎么对待{who}"
            "（例如「最近压力很大，多关心少打趣」），要有记忆依据。"
        )
    user = f"【现有卡片】\nprofile：{card.get('profile') or '（空）'}\n现有事实：\n{facts}\n【新的记忆】\n" + "\n".join(
        f"- {m}" for m in memories
    )
    if recap:
        user += f"\n【最近一次聊天的前情】{recap}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def self_messages(bot_name: str, persona: str, previous: str, memories: list[str]) -> list[dict]:
    """让模型概括她自己最近的生活。"""
    system = (
        f"你在帮陪伴角色「{bot_name}」写一句「最近的生活」，{SELF_MAX} 字以内，第二人称（「你最近…」），"
        "写她最近在做什么、心情和生活里的变化。只根据给出的经历，只输出这句话。"
    )
    user = f"【角色设定】{persona[:200]}\n【上次写的】{previous or '（无）'}\n【最近的经历】\n" + "\n".join(f"- {m}" for m in memories)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
