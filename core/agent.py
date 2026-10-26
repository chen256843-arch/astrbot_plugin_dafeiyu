"""智能体编排：把记忆、驱力、环境、经济、社交串成一个长期运行的角色。

与具体机器人框架无关，便于单独测试；插件层只负责收发消息与定时调度。
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import gua, sun, weather
from .collapse import collapse, pick
from .drives import DRIVES, Drives
from .ecology import CROPS, Ecology
from .economy import INCOME, Economy
from .embedding import Embedder
from .llm import LLM
from .memory import REGIONS, MemoryStore, affect
from .social import BOUNDARY, Social
from .util import DAY, HOUR, atomic_write_json, now, read_json
from .world import LANDMARKS, World

log = logging.getLogger("dafeiyu.agent")

BOOKS = ("小王子", "瓦尔登湖", "昆虫记", "植物的秘密生活", "星空的故事", "厨房里的化学", "猫的哲学", "海边的卡夫卡")
ATE_PATTERN = re.compile(r"吃完了|吃饱了|吃过了|吃好了|刚吃完")
PRAISE_PATTERN = re.compile(r"真棒|好厉害|厉害|好乖|乖|夸夸|谢谢你")
JOB_INTERVALS = {
    "tick": 600,
    "weather": 3600,
    "ingest": 600,
    "maintain": 3600,
    "consolidate": DAY,
    "diary": DAY,
    "selftest": 6 * HOUR,
    "save": 300,
}


@dataclass
class Settings:
    data_dir: Path
    name: str = "大肥鱼"
    aliases: list[str] = field(default_factory=list)
    persona: str = "一个住在浮空小岛屋子里的女孩，养了一只叫小灰的猫，种菜、看书、照顾阳台的花。说话温柔，有点小迷糊。"
    group_persona: str = ""  # 仅群聊使用；空则沿用 persona
    owners: set[str] = field(default_factory=set)
    owner_nick: str = ""
    interests: list[str] = field(default_factory=lambda: ["猫", "书", "花", "种菜", "天气", "海"])
    home: str = "海"
    group_p0: float = 0.03
    proactive: bool = True
    quiet_start: int = 23
    quiet_end: int = 8
    embed_api_key: str = ""
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_model: str = "text-embedding-v4"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    price_in: float = 0.002  # 每千 token（元），按模型官网价格调整
    price_out: float = 0.008
    hot_capacity: int = 5000
    tz: str = "Asia/Shanghai"


@dataclass
class Incoming:
    uid: str
    nick: str
    text: str
    group_id: str | None = None
    is_tome: bool = False


class Agent:
    def __init__(self, settings: Settings, rng: random.Random | None = None):
        self.s = settings
        self.rng = rng or random.Random()
        self.drives = Drives()
        self.memory = MemoryStore(capacity=settings.hot_capacity, rng=self.rng)
        self.world = World(self.rng)
        self.eco = Ecology(self.rng)
        self.econ = Economy()
        self.social = Social(set(settings.owners), settings.owner_nick)
        self.embedder = Embedder(settings.embed_api_key, settings.embed_base_url, settings.embed_model)
        self.llm = LLM(settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.price_in, settings.price_out)
        try:
            self.tz = ZoneInfo(settings.tz)
        except Exception:  # noqa: BLE001  容器里缺时区数据时退回东八区
            log.warning("找不到时区 %s，改用 UTC+8", settings.tz)
            self.tz = timezone(timedelta(hours=8))
        self.host: dict = {}  # 宿主框架自用的持久化数据（如主人的会话标识）
        self.weather: dict[str, dict] = {}
        self.events: list[dict] = []
        self.diaries: list[dict] = []
        self.activity: dict = {"name": "发呆", "until": 0.0, "basis": "安静"}
        self.body = {"hunger": 30.0}
        self.health: dict[str, float] = {}
        self.last_tick = now()
        self.last_owner_msg = 0.0
        self.last_proactive = 0.0
        self.history: dict[str, list[dict]] = {}

    # ================================================================ 持久化
    @property
    def _paths(self) -> tuple[Path, Path, Path]:
        d = self.s.data_dir
        return d / "state.json", d / "memory.json", d / "vectors.json"

    def load(self) -> None:
        sp, mp, vp = self._paths
        st = read_json(sp, {})
        self.memory = MemoryStore.from_dict(read_json(mp, None), read_json(vp, {}), capacity=self.s.hot_capacity, rng=self.rng)
        self.drives = Drives.from_dict(st.get("drives"))
        self.world = World.from_dict(st.get("world"), self.rng)
        self.eco = Ecology.from_dict(st.get("eco"), self.rng)
        self.econ = Economy.from_dict(st.get("econ"))
        self.social.load(st.get("social"))
        self.llm.load(st.get("llm"))
        self.weather = st.get("weather", {})
        self.events = st.get("events", [])
        self.diaries = st.get("diaries", [])
        self.activity = st.get("activity", self.activity)
        self.body = st.get("body", self.body)
        self.health = st.get("health", {})
        self.last_tick = st.get("last_tick", now())
        self.last_owner_msg = st.get("last_owner_msg", 0.0)
        self.last_proactive = st.get("last_proactive", 0.0)
        self.host = st.get("host", {})

    def save(self) -> None:
        sp, mp, vp = self._paths
        self.ok("save")
        atomic_write_json(
            sp,
            {
                "drives": self.drives.to_dict(),
                "world": self.world.to_dict(),
                "eco": self.eco.to_dict(),
                "econ": self.econ.to_dict(),
                "social": self.social.to_dict(),
                "llm": self.llm.to_dict(),
                "weather": self.weather,
                "events": self.events[-200:],
                "diaries": self.diaries[-30:],
                "activity": self.activity,
                "body": self.body,
                "health": self.health,
                "last_tick": self.last_tick,
                "last_owner_msg": self.last_owner_msg,
                "last_proactive": self.last_proactive,
                "host": self.host,
            },
        )
        atomic_write_json(mp, self.memory.to_dict())
        atomic_write_json(vp, self.memory.vectors_dict())

    def _dt(self) -> datetime:
        """角色所在时区的当前时间（容器默认 UTC，作息不能跟着它走）。"""
        return datetime.now(self.tz)

    def ok(self, job: str) -> None:
        self.health[job] = now()

    # ================================================================ 小工具
    def _event(self, text: str, memorable: bool = False, tags: tuple[str, ...] = ()) -> None:
        self.events.append({"ts": now(), "text": text})
        self.events = self.events[-200:]
        if memorable:
            self.memory.propose(text, source="self", tags=["自主", *tags])

    def _home_weather(self) -> dict:
        w = self.weather.get(self.s.home)
        if not w:
            w = self.weather[self.s.home] = weather.simulate(None, self.rng)
        return w

    def _sun(self) -> tuple[float, float]:
        lat, lon = weather.LOCATIONS.get(self.s.home, (30.0, 120.0))
        sh = sun.sin_elevation(lat, lon, datetime.now(timezone.utc))
        return sh, sun.irradiance(sh, self._home_weather().get("cloud", 0.5))

    def _quiet(self, hour: int) -> bool:
        a, b = self.s.quiet_start, self.s.quiet_end
        return hour >= a or hour < b if a > b else a <= hour < b

    def mood_word(self) -> str:
        m = self.drives.mood
        return "很好" if m > 0.5 else "不错" if m > 0.15 else "平静" if m > -0.15 else "有点低落" if m > -0.5 else "难过"

    def location_text(self) -> str:
        loc = self.world.her_location()
        return f"屋子的{self.world.her['room']}" if loc == "屋子" else loc

    # ================================================================ 自主节律
    async def job_tick(self) -> list[tuple[str, str]]:
        """每 10 分钟：环境推进 + 规则引擎决策。返回需要主动发送的 (uid, text)。"""
        t = now()
        dt_h = min(6.0, max(0.0, (t - self.last_tick) / HOUR))
        self.last_tick = t
        hour = self._dt().hour
        w = self._home_weather()
        sh, irr = self._sun()
        asleep = self.activity["name"] == "睡觉"
        self.drives.tick(dt_h, asleep=asleep)
        self.body["hunger"] = min(100.0, self.body["hunger"] + (1.5 if asleep else 4.0) * dt_h)

        self.world.tick_rooms(dt_h, w.get("temp", 22), irr, warmth=self.econ.bonus("warmth"))
        for e in self.world.tick_agents(dt_h, daylight=sh > 0):
            self._event(e)
        gh = int(self.econ.bonus("greenhouse"))
        for e in self.eco.tick_farm(dt_h, w, greenhouse=gh, growth_bonus=self.econ.bonus("growth")):
            self._event(e, memorable=True, tags=("菜",))
        for e in self.eco.tick_flowers(dt_h, irr):
            self._event(e, memorable=True, tags=("花",))
        if sh > 0 and self.world.at("鸟", "菜棚"):
            e = self.eco.bird_peck(dt_h, gh)
            if e:
                self._event(e)
        self.eco.tick_bugs(dt_h, self._dt().month, w, at_sea=self.s.home == "海")
        cat = self.world.agents["小灰"]
        if self.world.at("小灰", "菜棚", 8) or self.world.at("小灰", "屋子", 8):
            e = self.eco.cat_hunt(cat)
            if e:
                self._event(e, memorable=self.rng.random() < 0.3, tags=("猫",))
        e = self.eco.small_animals(dt_h)
        if e:
            self._event(e)
        if self.eco.spoil(t):
            self._event("有饭放太久，变成了剩饭")
        if self.eco.blooming() and self.world.her_location() in ("阳台", "屋子"):
            self.drives.feel(0.02 * dt_h * self.eco.blooming(), weight=1.0)

        out: list[tuple[str, str]] = []
        if t >= self.activity.get("until", 0):
            msg = self._decide(t, hour)
            if msg:
                out.extend((uid, msg) for uid in self.s.owners)
        self.ok("tick")
        return out

    def _set_activity(self, name: str, hours: float, basis: str) -> None:
        self.activity = {"name": name, "until": now() + hours * HOUR, "basis": basis}

    def _decide(self, t: float, hour: int) -> str | None:
        self.llm.rule_actions += 1
        if hour >= 23 or hour < 7:
            if self.activity["name"] != "睡觉":
                self.world.go("屋子")
                self.world.go_room("卧室")
                self._event("她回卧室睡觉了")
            self._set_activity("睡觉", 1.0, "安静")
            return None
        if self.activity["name"] == "睡觉":
            self._event("她起床了")
            self.eco.daily_taste()
        if self.body["hunger"] > 70:
            if self._eat_or_cook():
                return None
        basis = self.drives.decide(hour, self.rng).choice
        handler = {"读": self._do_read, "玩": self._do_play, "唤醒": self._do_reach, "安静": self._do_quiet, "学": self._do_work}
        return handler[basis](hour)

    def _eat_or_cook(self) -> bool:
        dish = self.eco.eat()
        if dish:
            self.body["hunger"] = max(0.0, self.body["hunger"] - 50)
            self._set_activity("吃饭", 0.5, "安静")
            self._event(f"她吃了{dish}")
            return True
        if not self.eco.feasible() and self.econ.coins >= 1:
            self.econ.buy_goods("米", 1, self.eco.pantry)
        dish = self.eco.cook(now())
        if dish:
            self.world.go_room("厨房")
            self._set_activity("做饭", 0.75, "学")
            self._event(f"她在厨房做了{dish}（今天想吃{self.eco.taste}的）")
            return True
        return False

    def _do_read(self, hour: int) -> None:
        self.world.go_room("书房")
        book = pick({b: 1.0 for b in BOOKS}, self.rng)
        coins = INCOME["读书"] + int(self.econ.bonus("read"))
        self.econ.earn(coins, "读书")
        self.drives.satisfy("curiosity", 18)
        self.drives.satisfy("achievement", 6)
        self._set_activity(f"读《{book}》", self.rng.uniform(0.75, 1.5), "读")
        self._event(f"在书房读了一会儿《{book}》", memorable=self.rng.random() < 0.25, tags=("书",))

    def _do_play(self, hour: int) -> None:
        raining = self._home_weather().get("precipitation", 0) > 0.2
        if raining or hour >= 19:
            place = "屋子"
        else:
            place = pick({"码头": 1.0, "老树": 1.0, "水井": 0.6, "崖边": 0.5, "阳台": 0.8}, self.rng)
        cost = self.world.go(place) if place != "屋子" else 0.0
        if place == "屋子":
            self.world.go_room("客厅")
        self.drives.bump("rest", cost)
        self.drives.satisfy("play", 20)
        cat = self.world.agents["小灰"]
        if cat["mode"] == "跟随" or place == "屋子":
            cat["intimacy"] = min(100.0, cat["intimacy"] + 2 + 4 * self.econ.bonus("cat"))
            self.drives.satisfy("social", 5)
            self._event(f"和小灰在{place}玩了一会儿", memorable=self.rng.random() < 0.2, tags=("猫",))
        else:
            self._event(f"一个人去{place}逛了逛")
        self._set_activity(f"在{place}玩", self.rng.uniform(0.5, 1.2), "玩")

    def _do_reach(self, hour: int) -> str | None:
        t = now()
        self.drives.satisfy("social", 10)
        can = (
            self.s.proactive
            and self.s.owners
            and not self._quiet(hour)
            and t - self.last_proactive > 3 * HOUR
            and t - self.last_owner_msg > 2 * HOUR
        )
        self._set_activity("想主人", 0.3, "唤醒")
        if not can:
            self._event("有点想主人了")
            return None
        self.last_proactive = t
        recent = [e["text"] for e in self.events[-5:]]
        last = recent[-1] if recent else "在发呆"
        self._event("给主人发了条消息")
        return pick({f"在干嘛呀？我刚{last}。": 1.0, f"突然想你了～{last}。": 0.6, "今天过得怎么样？": 0.6}, self.rng)

    def _do_quiet(self, hour: int) -> None:
        if self.drives.napping:
            self.world.go_room("卧室")
            self._set_activity("打盹", 1.0, "安静")
            self._event("困了，在卧室打了个盹")
        elif self.eco.blooming() and hour < 19:
            self.world.go("阳台")
            self._set_activity("在阳台看花", 0.6, "安静")
            self._event("在阳台看花发呆")
        else:
            self.world.go_room("客厅")
            self._set_activity("发呆", 0.5, "安静")
        self.drives.satisfy("rest", 20)

    def _do_work(self, hour: int) -> None:
        eco, econ = self.eco, self.econ
        cat = self.world.agents["小灰"]
        opts = {"学习": 0.3}
        if any(p["crop"] and p["water"] < 40 for p in eco.plots) or any(p["water"] < 30 for p in eco.pots):
            opts["浇水"] = 1.0
        if any(p["crop"] and p["growth"] >= 100 for p in eco.plots):
            opts["收菜"] = 1.5
        if any(not p["crop"] for p in eco.plots) and econ.coins >= 5:
            opts["种地"] = 0.7
        if self.world.rooms[self.world.dustiest()]["dust"] > 30:
            opts["家务"] = 0.8
        if cat["hunger"] > 50:
            opts["喂猫"] = 1.2
        if eco.blooming():
            opts["剪花"] = 0.5
        if hour < 18:
            opts["拾荒"] = 0.4
        chore = collapse(opts, self.rng).choice
        text = self.chore(chore)
        self.drives.satisfy("achievement", 10)
        self._set_activity(chore, self.rng.uniform(0.5, 1.0), "学")
        self._event(text, memorable=chore in ("收菜", "拾荒") and "金币" in text)

    def chore(self, name: str, arg: str | None = None) -> str:
        """家务/农活的具体执行，命令也复用这里。"""
        eco, econ, world = self.eco, self.econ, self.world
        cat = world.agents["小灰"]
        if name == "浇水":
            world.go("菜棚")
            eco.water()
            return "给菜地和阳台的花浇了水"
        if name == "收菜":
            got = [r for i in range(4) if (r := eco.harvest(i))]
            if not got:
                return "菜还没熟"
            earned = 0
            for crop, n in got:
                sell = n // 2
                eco.pantry[crop] -= sell
                earned += sell * CROPS[crop]["price"]
            if earned:
                econ.earn(earned, "收菜")
            return "收了" + "、".join(f"{c}×{n}" for c, n in got) + (f"，卖掉一半得了 {earned} 金币" if earned else "")
        if name == "种地":
            idx = next((i for i, p in enumerate(eco.plots) if not p["crop"]), None)
            if idx is None:
                return "地都种满了"
            crop = arg if arg in CROPS else pick({c: 1.0 for c in CROPS}, self.rng)
            if not econ.spend(CROPS[crop]["seed"], f"{crop}种子"):
                return "买种子的钱不够"
            p = eco.plant(idx, crop)
            return f"在{idx + 1}号地种下了{crop}（{p['variety']}品种）"
        if name == "家务":
            room = world.dustiest()
            world.go_room(room)
            world.clean(room)
            econ.earn(INCOME["家务"], "家务")
            return f"把{room}打扫干净了"
        if name == "喂猫":
            if eco.pantry.get("猫粮", 0) <= 0:
                if not econ.spend(3, "猫粮"):
                    return "猫粮没了，也没钱买"
                eco.pantry["猫粮"] = 1
            eco.pantry["猫粮"] -= 1
            cat["hunger"] = max(0.0, cat["hunger"] - 40)
            cat["intimacy"] = min(100.0, cat["intimacy"] + 3)
            econ.earn(INCOME["喂猫"], "喂猫")
            return "喂了小灰，它蹭了蹭她的手"
        if name == "剪花":
            kind = eco.cut_flower()
            if not kind:
                return "还没有开好的花"
            econ.earn(INCOME["剪花"], "剪花")
            return f"剪了一枝{kind}，换了 {INCOME['剪花']} 金币"
        if name == "拾荒":
            place = pick({"码头": 1.0, "崖边": 0.7}, self.rng)
            self.drives.bump("rest", world.go(place))
            found = int(pick({"0": 0.3, "2": 0.35, "5": 0.25, "10": 0.1}, self.rng))
            if found:
                econ.earn(found, "拾荒")
                return f"在{place}捡到些东西，换了 {found} 金币"
            return f"去{place}转了一圈，什么也没捡到"
        self.world.go_room("书房")
        self.drives.satisfy("curiosity", 5)
        return "在书房学了点新东西"

    # ================================================================ 其他定时任务
    async def job_weather(self) -> None:
        self.weather = await weather.update_all(self.weather, self.rng)
        self.ok("weather")

    async def job_ingest(self, batch: int = 50) -> dict:
        written = self.memory.judge()
        backend = self.embedder.backend
        missing = self.memory.missing_vectors(backend)[:batch]
        if missing:
            texts = [self.memory.nodes[i]["text"] for i in missing]
            used, vecs = await self.embedder.embed(texts)
            for nid, v in zip(missing, vecs):
                self.memory.set_vector(nid, used, v)
        self.ok("ingest")
        return {"written": len(written), "embedded": len(missing)}

    def job_maintain(self) -> dict:
        fixed = self.memory.maintain()
        self.ok("maintain")
        return fixed

    async def job_consolidate(self) -> dict:
        summaries: dict[str, str] = {}
        if self.llm.enabled:
            for tag, ids in self.memory.abstraction_plans():
                texts = "\n".join(f"- {self.memory.nodes[i]['text']}" for i in ids[:8] if i in self.memory.nodes)
                s = await self.llm.chat(
                    [
                        {"role": "system", "content": "把下面几条记忆抽象成一句简短的概括（30 字以内），只输出这句话。"},
                        {"role": "user", "content": f"主题：{tag}\n{texts}"},
                    ],
                    max_tokens=80,
                    temperature=0.3,
                )
                if s:
                    summaries[tag] = s
        report = self.memory.consolidate(lambda tag, ids: summaries.get(tag) or self.memory.template_abstraction(tag, ids))
        self.ok("consolidate")
        return report

    async def job_diary(self) -> str:
        t = now()
        today = [e["text"] for e in self.events if t - e["ts"] < DAY]
        if not today:
            today = ["平静的一天"]
        text = None
        if self.llm.enabled:
            text = await self.llm.chat(
                [
                    {"role": "system", "content": f"你是{self.s.name}。{self.s.persona} 用第一人称写一篇 100 字以内的日记，口语化，不要标题。"},
                    {"role": "user", "content": "今天发生的事：\n" + "\n".join(today[-30:])},
                ],
                max_tokens=220,
            )
        if not text:
            picks = today[-3:]
            text = f"今天{'，'.join(picks)}。心情{self.mood_word()}。"
        self.diaries.append({"date": self._dt().strftime("%Y-%m-%d"), "text": text})
        self.memory.write(text, source="self", region="episodic", tags=["日记"])
        self.ok("diary")
        return text

    def job_selftest(self) -> list[str]:
        problems = []
        names = {gua.encode_bits([(i >> b) & 1 for b in range(6)]) for i in range(64)}
        if len(names) != 64:
            problems.append(f"卦象编码不唯一：{len(names)}/64")
        if gua.encode_bits([1] * 6) != "乾为天" or gua.encode_bits([0] * 6) != "坤为地":
            problems.append("乾/坤编码错误")
        probs = collapse({"a": 0.3, "b": 0.2, "c": 0.5}, self.rng).probs
        if abs(sum(probs.values()) - 1) > 1e-6:
            problems.append("玻恩概率未归一")
        if any(not 0 <= self.drives.value[d] <= 100 for d in DRIVES):
            problems.append("驱力越界")
        fixed = self.memory.maintain()
        if fixed:
            problems.append(f"图结构被修复：{fixed}")
        self.ok("selftest")
        return problems

    def health_report(self) -> list[tuple[str, float, bool]]:
        t = now()
        out = []
        for job, interval in JOB_INTERVALS.items():
            lag = t - self.health.get(job, 0)
            out.append((job, lag, lag <= 2 * interval))
        return out

    # ================================================================ 对话
    async def on_message(self, msg: Incoming) -> str | None:
        uid, text = str(msg.uid), msg.text.strip()
        if not text:
            return None
        is_owner = self.social.is_owner(uid)
        valence, _ = affect(text)
        self.social.check_impersonation(uid, msg.nick)
        self.social.touch(uid, msg.nick, valence)
        tags = [f"user:{uid}"] + ([f"group:{msg.group_id}"] if msg.group_id else [])

        if msg.group_id:
            p, r, bypass = self.social.trigger(
                text, msg.group_id, msg.is_tome, is_owner, self.s.interests,
                [self.s.name, *self.s.aliases], self.drives.value["social"], self.s.group_p0,
            )
            if not bypass and self.rng.random() >= p:
                if r >= 2:
                    self.memory.propose(f"{msg.nick}在群里说：{text}", source="owner" if is_owner else "other", tags=tags)
                return None
            self.social.mark_replied(msg.group_id)

        if is_owner:
            self.last_owner_msg = now()
            if not msg.group_id and ATE_PATTERN.search(text):
                self._feed_link()

        speaker = "主人" if is_owner else msg.nick or uid
        self.memory.propose(f"{speaker}说：{text}", source="owner" if is_owner else "other", tags=tags)

        reply = await self._reply(msg, is_owner)
        self.drives.satisfy("social", 15 if is_owner else 6)
        if PRAISE_PATTERN.search(text):
            self.drives.satisfy("recognition", 15)
            valence = max(valence, 0.4)
        self.drives.feel(valence, 0.25 if is_owner else 0.1)
        return reply

    def _feed_link(self) -> None:
        """投喂联动：主人说吃完了 → 她也吃掉一份餐，记一条事件。"""
        dish = self.eco.eat()
        if dish:
            self.body["hunger"] = max(0.0, self.body["hunger"] - 50)
            self._event(f"主人吃完饭了，她也跟着吃了一份{dish}", memorable=True, tags=("饭",))
        else:
            self._event("主人吃完饭了，可她还没做饭")

    async def _reply(self, msg: Incoming, is_owner: bool) -> str:
        uid = str(msg.uid)
        _, vecs = await self.embedder.embed([msg.text])
        temp_label, temp = self.drives.recall_temperature(self.rng)
        hits = self.memory.retrieve(
            msg.text,
            k=6,
            query_vec=vecs[0] if vecs else None,
            backend=self.embedder.backend,
            drive_bias=self.drives.drive_bias,
            mood=self.drives.mood,
            temperature=temp,
        )
        if msg.group_id:
            hits = [h for h in hits if self._visible_in_group(h.node, uid)]

        key = f"g:{msg.group_id}" if msg.group_id else f"p:{uid}"
        hist = self.history.setdefault(key, [])
        speaker = msg.nick or uid
        user_line = msg.text if not msg.group_id else f"{speaker}：{msg.text}"

        reply = None
        if self.llm.enabled:
            messages = [{"role": "system", "content": self._system_prompt(uid, msg, hits)}, *hist[-10:], {"role": "user", "content": user_line}]
            reply = await self.llm.chat(messages)
            if reply:
                self.llm.llm_actions += 1
        if not reply:
            reply = self._template_reply(msg.text, hits)
        hist.extend([{"role": "user", "content": user_line}, {"role": "assistant", "content": reply}])
        del hist[:-20]
        return reply

    @staticmethod
    def _visible_in_group(node: dict, uid: str) -> bool:
        """群聊里只用：群聊记忆、自主记忆、或说话人自己的记忆；别人的私聊不外传。"""
        tags = node.get("tags", [])
        if any(t.startswith("group:") for t in tags):
            return True
        users = [t for t in tags if t.startswith("user:")]
        return not users or f"user:{uid}" in users

    def _system_prompt(self, uid: str, msg: Incoming, hits) -> str:
        tier = self.social.tier(uid)
        rel = self.social.rel.get(uid, {})
        w = self._home_weather()
        cat = self.world.agents["小灰"]
        mem_lines = "\n".join(
            f"- [{REGIONS.get(h.node['region'], h.node['region'])}] {h.node['text']}（{datetime.fromtimestamp(h.node['created'], self.tz).strftime('%m-%d')}）"
            for h in hits
        ) or "（没有想起什么相关的）"
        taboos = "、".join(rel.get("taboos", [])) or "无"
        where = "群聊" if msg.group_id else "私聊"
        persona = self.s.group_persona if (msg.group_id and self.s.group_persona) else self.s.persona
        return (
            f"你是「{self.s.name}」。{persona}\n"
            f"【此刻】{self._dt().strftime('%H:%M')}，{weather.describe(w)}。你在{self.location_text()}，正在{self.activity['name']}。"
            f"心情{self.mood_word()}。驱力：{self.drives.summary()}。小灰{cat['mode']}中。金币 {self.econ.coins}。\n"
            f"【对方】{msg.nick or uid}（{where}），关系：{tier}。相处边界：{BOUNDARY[tier]}。禁忌话题：{taboos}\n"
            f"【想起的记忆】\n{mem_lines}\n"
            "【说话方式】像发 QQ 消息一样口语化，一般 1–3 句，不用 markdown 和列表。只根据上面的记忆谈过去的事，记不清就直说。"
            "如果对方认真地问你是不是 AI、有没有意识，要诚实说明你是由程序驱动的角色，没有真实的主观感受。"
        )

    def _template_reply(self, text: str, hits) -> str:
        base = f"我在{self.location_text()}{self.activity['name']}呢"
        if hits:
            return f"{base}。我记得：「{hits[0].node['text'][:40]}」"
        return pick({f"{base}～": 1.0, f"嗯嗯，{base}。": 1.0, "诶？怎么啦": 0.5}, self.rng)

    # ================================================================ 命令用的文本
    def status_text(self) -> str:
        w = self._home_weather()
        _, irr = self._sun()
        lat, lon = weather.LOCATIONS.get(self.s.home, (30.0, 120.0))
        dl = sun.day_length(lat, self._dt().timetuple().tm_yday)
        elev = sun.elevation_deg(lat, lon, datetime.now(timezone.utc))
        cat = self.world.agents["小灰"]
        room = self.world.rooms[self.world.her["room"]]
        light = self.world.light(self.world.her["room"], irr, self.econ.bonus("light"))
        return "\n".join(
            [
                f"📍 {self.location_text()} · {self.activity['name']}",
                f"🌤 {self.s.home}：{weather.describe(w)}{'（模拟）' if w.get('simulated') else ''}",
                f"☀ 太阳高度 {elev:.0f}° 昼长 {dl:.1f}h 室内光照 {light:.2f} 室温 {room['temp']:.1f}°C",
                f"💗 心情{self.mood_word()}（{self.drives.mood:+.2f}）饱腹 {100 - self.body['hunger']:.0f}{' 💤打盹中' if self.drives.napping else ''}",
                f"⚡ {self.drives.summary()}",
                f"🐱 小灰 {cat['mode']} 饥饿{cat['hunger']:.0f} 亲密{cat['intimacy']:.0f} · 🐦 鸟 {self.world.agents['鸟']['mode']}",
                f"💰 {self.econ.coins} 金币 · 🍚 现成饭 {len(self.eco.meals)} 份 · 剩饭 {self.eco.leftovers} · 虫 {self.eco.bugs:.0f}",
            ]
        )

    def farm_text(self) -> str:
        lines = []
        for i, p in enumerate(self.eco.plots):
            if p["crop"]:
                lines.append(f"{i + 1}号地 {p['crop']}({p['variety']}) 生长{p['growth']:.0f} 水{p['water']:.0f} 健康{p['health']:.0f}")
            else:
                lines.append(f"{i + 1}号地 空")
        pots = []
        for p in self.eco.pots:
            pots.append(p["kind"] + ("🌸" if p["bloom"] >= 100 else "%.0f" % p["bloom"]))
        lines.append("阳台：" + " ".join(pots))
        lines.append("温室棚等级 %d · 生长加成 +%.0f%%" % (int(self.econ.bonus("greenhouse")), self.econ.bonus("growth") * 100))
        lines.append("库存：" + (" ".join(f"{k}{v}" for k, v in self.eco.pantry.items() if v) or "空"))
        return "\n".join(lines)

    def memory_text(self, query: str) -> str:
        hits = self.memory.retrieve(query, k=6, drive_bias=self.drives.drive_bias, mood=self.drives.mood, update=False)
        if not hits:
            return "没有想起相关的记忆"
        lines = []
        for h in hits:
            n = h.node
            bits = gua.bits_str(gua.node_bits(n, self.memory.degree(n["id"])))
            lines.append(
                f"{n['id']} [{REGIONS[n['region']]}/{n['tier']}] {h.gua}({bits}) 分{h.score:.2f} "
                f"cos{h.parts.get('cos', 0):.2f} lex{h.parts.get('lex', 0):.2f} w{h.parts.get('w', 0):.2f}\n  {n['text'][:60]}"
            )
        return "\n".join(lines)

    def memory_stats_text(self) -> str:
        st = self.memory.stats()
        regions = " ".join(f"{REGIONS.get(k, k)}{v}" for k, v in st["regions"].items())
        return f"热库 {st['hot']}/{self.memory.capacity} 冷库 {st['cold']} 边 {st['edges']} 向量 {st['vectors']} 候选池 {st['pool']}\n分区：{regions}"

    def remember(self, text: str, uid: str) -> str:
        n = self.memory.write(text, source="owner" if self.social.is_owner(uid) else "other", tier="core", tags=[f"user:{uid}"])
        return f"记住了（{n['id']}，{REGIONS[n['region']]}，{self.memory.gua_of(n['id'])}）"
