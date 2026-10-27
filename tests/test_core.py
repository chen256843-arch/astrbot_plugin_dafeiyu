"""核心算法测试（不依赖 AstrBot）。运行：python -m unittest discover -s tests"""
import asyncio
import json
import math
import random
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import gua, sun  # noqa: E402
from core.agent import HIST_KEEP, HIST_MAX, Agent, Incoming, Settings  # noqa: E402
from core.bank import SELF, BankHit, MemoryBank, time_label  # noqa: E402
from core.collapse import born, collapse  # noqa: E402
from core.drives import Drives  # noqa: E402
from core.ecology import Ecology  # noqa: E402
from core.economy import Economy  # noqa: E402
from core.embedding import hash_embed  # noqa: E402
from core.memory import DAY, Hit, MemoryStore  # noqa: E402
from core.social import Social  # noqa: E402
from core.world import World  # noqa: E402


class TestGua(unittest.TestCase):
    def test_64_unique(self):
        names = {gua.encode_bits([(i >> b) & 1 for b in range(6)]) for i in range(64)}
        self.assertEqual(len(names), 64)

    def test_qian_kun(self):
        self.assertEqual(gua.encode_bits([1] * 6), "乾为天")
        self.assertEqual(gua.encode_bits([0] * 6), "坤为地")
        # 下乾上坤 = 地天泰；下坤上乾 = 天地否
        self.assertEqual(gua.encode_bits([1, 1, 1, 0, 0, 0]), "地天泰")
        self.assertEqual(gua.encode_bits([0, 0, 0, 1, 1, 1]), "天地否")


class TestCollapse(unittest.TestCase):
    def test_born_normalized(self):
        p = born({"a": 1 + 1j, "b": 0.5, "c": 0})
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertAlmostEqual(p["a"] / p["b"], 2 / 0.25)

    def test_deterministic_first(self):
        c = collapse({"读": 0.8, "玩": 0.1, "学": 0.1}, random.Random(0))
        self.assertEqual((c.choice, c.mode), ("读", "deterministic"))

    def test_competing_sampled(self):
        rng = random.Random(1)
        modes = {collapse({"读": 0.5, "玩": 0.45}, rng).mode for _ in range(20)}
        self.assertEqual(modes, {"sampled"})

    def test_interference(self):
        # 同相增强、反相相消
        same = collapse({"a": 1, "b": 1}, context={"a": 1}, phase={"a": 0}).probs["a"]
        anti = collapse({"a": 1, "b": 1}, context={"a": 1}, phase={"a": math.pi}).probs["a"]
        self.assertGreater(same, 0.75)
        self.assertLess(anti, 0.05)


class TestDrives(unittest.TestCase):
    def test_inertia(self):
        d = Drives()
        d.bump("play", 50)
        d.tick(0.0)
        # v = 0.7·40 + 0.3·90 = 55
        self.assertAlmostEqual(d.value["play"], 55.0)

    def test_nap(self):
        d = Drives()
        d.value["rest"] = d.target["rest"] = 85
        d.tick(0.1)
        self.assertTrue(d.napping)
        for _ in range(30):
            d.tick(0.5)
        self.assertFalse(d.napping)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.m = MemoryStore(rng=random.Random(0))

    def test_write_weight(self):
        n = self.m.write("主人说下周三要去海边看日出", source="owner")
        self.assertAlmostEqual(n["surprise"], 1.0)
        self.assertAlmostEqual(n["w0"], 8.0)  # 4.0 × (1 + 1)
        n2 = self.m.write("主人说下周三要去海边看日落", source="other")
        self.assertLess(n2["surprise"], 0.3)

    def test_decay(self):
        n = self.m.write("今天在书房看书", source="other", tier="normal")
        t = n["last_access"] + 7 * DAY
        self.assertAlmostEqual(self.m.weight(n, t), n["w0"] * math.exp(-1), places=5)

    def test_flash(self):
        n = self.m.write("太害怕了！哭了好久！崩溃！", source="owner")
        self.assertEqual(n["region"], "flash")
        self.assertEqual(n["tier"], "core")

    def test_judge_filters(self):
        self.m.propose("哈哈")
        self.m.propose("小灰今天在菜棚边抓到一只虫子")
        self.m.propose("小灰今天在菜棚边抓到一只虫子")
        written = self.m.judge()
        self.assertEqual(len(written), 1)

    def test_retrieve_lexical_and_vector(self):
        texts = ["主人喜欢喝咖啡不加糖", "小灰是一只灰色的猫", "菜棚里种了番茄", "主人的生日是三月五号"]
        for t in texts:
            n = self.m.write(t, source="owner")
            self.m.set_vector(n["id"], "hash512", hash_embed(t))
        q = "主人生日是哪天"
        hits = self.m.retrieve(q, k=2, query_vec=hash_embed(q), backend="hash512")
        self.assertIn("生日", hits[0].node["text"])
        self.assertTrue(hits[0].gua)

    def test_dedupe_and_flash_exempt(self):
        a = self.m.write("我们一起去码头看了很久的海浪和夕阳", source="other", region="episodic")
        b = self.m.write("我们一起去码头看了很久的海浪和夕阳呀", source="other", region="episodic")
        self.m.write("救命！太害怕了！哭了！", source="owner")
        self.m.write("救命！太害怕了！哭了！！", source="owner")
        merged = self.m.dedupe()
        self.assertEqual(merged, 1)
        self.assertEqual(sum(1 for n in self.m.nodes.values() if n["region"] == "flash"), 2)
        self.assertTrue((a["id"] in self.m.nodes) ^ (b["id"] in self.m.nodes))

    def test_forget_and_capacity(self):
        m = MemoryStore(capacity=3, rng=random.Random(0))
        for i in range(5):
            m.write(f"第{i}条完全不同的记忆内容{'甲乙丙丁戊'[i] * 4}", source="other")
        self.assertEqual(len(m.nodes), 3)
        self.assertEqual(len(m.cold), 2)

    def test_maintain(self):
        a = self.m.write("记忆A关于猫", source="other")
        self.m.edges[a["id"]][a["id"]] = 1.0
        self.m.edges[a["id"]]["m999"] = 0.5
        self.m.vectors["m998"] = {"b": "x", "v": [1.0]}
        fixed = self.m.maintain()
        self.assertEqual(fixed, {"self_loop": 1, "dangling": 1, "extra_vector": 1})
        self.assertEqual(self.m.maintain(), {})

    def test_roundtrip(self):
        self.m.write("主人喜欢草莓", source="owner")
        m2 = MemoryStore.from_dict(self.m.to_dict(), self.m.vectors)
        self.assertEqual(m2.nodes.keys(), self.m.nodes.keys())
        self.assertTrue(m2.retrieve("草莓"))

    def test_consolidate_abstraction(self):
        for t in ("今天小灰在屋子里睡觉", "昨天小灰抓了一只虫", "小灰今天跟着她去了码头"):
            self.m.write(t, source="other", region="episodic", tags=["猫"])
        r = self.m.consolidate()
        self.assertGreaterEqual(r["abstract"], 1)
        self.assertTrue(any(n["region"] == "semantic" for n in self.m.nodes.values()))


class TestSunWorld(unittest.TestCase):
    def test_day_length(self):
        self.assertAlmostEqual(sun.day_length(0, 80), 12.0, delta=0.1)
        self.assertEqual(sun.day_length(70, 172), 24.0)
        self.assertEqual(sun.day_length(70, 355), 0.0)

    def test_noon_elevation(self):
        # 春分前后正午，赤道上太阳接近天顶
        dt = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
        self.assertGreater(sun.elevation_deg(0, 0, dt), 85)

    def test_path_and_doors(self):
        w = World(random.Random(0))
        self.assertEqual(w.path("卧室", "厨房"), ["卧室", "客厅", "厨房"])
        w.set_door("厨房", "客厅", "locked")
        self.assertIsNone(w.path("卧室", "厨房"))
        self.assertLess(w.sound_level("客厅", "浴室"), w.sound_level("客厅", "卧室"))
        self.assertFalse(w.can_see("卧室", "浴室"))

    def test_move(self):
        w = World(random.Random(0))
        cost = w.go("码头")
        self.assertGreater(cost, 0)
        for _ in range(10):
            w.tick_agents(1 / 6, daylight=True)
        self.assertEqual(w.her_location(), "码头")


class TestEcoSocial(unittest.TestCase):
    def test_farm_grows(self):
        e = Ecology(random.Random(0))
        e.plant(0, "青菜")
        for _ in range(80):
            if e.plots[0]["water"] < 50:
                e.water(0, 20)
            e.tick_farm(1.0, {"temp": 20, "humidity": 60, "precipitation": 0, "wind": 5})
        self.assertIsNotNone(e.harvest(0))

    def test_furniture_cap(self):
        ec = Economy(coins=1000)
        for _ in range(3):
            ec.buy_furniture("生长灯")
        self.assertEqual(ec.furniture["生长灯"], 2)
        self.assertAlmostEqual(ec.bonus("growth"), 0.3)

    def test_trigger(self):
        s = Social({"1"}, "小鱼干")
        p_plain, _, _ = s.trigger("嗯", "g", False, False, ["猫"], ["大肥鱼"], 50)
        p_rich, _, _ = s.trigger("大肥鱼你家的猫怎么样了？最近还好吗", "g", False, False, ["猫"], ["大肥鱼"], 50)
        self.assertLess(p_plain, p_rich)
        self.assertLessEqual(p_rich, 0.85)
        self.assertEqual(s.trigger("x", "g", True, False, [], [], 0)[2], True)

    def test_impersonation(self):
        s = Social({"1"}, "小鱼干")
        self.assertTrue(s.check_impersonation("2", "小鱼干"))
        self.assertEqual(s.tier("2"), "疑似冒充")
        self.assertFalse(s.check_impersonation("1", "小鱼干"))
        self.assertEqual(s.tier("1"), "主人")


class TestSpeakerMemory(unittest.TestCase):
    """记住是谁说的。"""

    def setUp(self):
        self.m = MemoryStore(rng=random.Random(0))

    def test_same_words_different_people_both_kept(self):
        self.m.propose("我最喜欢的水果是草莓", tags=["user:1"], speaker="1", speaker_name="小明")
        self.m.propose("我最喜欢的水果是草莓", tags=["user:2"], speaker="2", speaker_name="小红")
        self.m.propose("我最喜欢的水果是草莓", tags=["user:1"], speaker="1", speaker_name="小明")
        written = self.m.judge()
        self.assertEqual(sorted(n["speaker"] for n in written), ["1", "2"])  # 第三条是小明重复，只强化
        self.assertEqual({n["speaker_name"] for n in written}, {"小明", "小红"})

    def test_dedupe_only_within_one_speaker(self):
        a = self.m.write("我们一起去码头看了很久的海浪和夕阳", region="episodic", speaker="1")
        b = self.m.write("我们一起去码头看了很久的海浪和夕阳呀", region="episodic", speaker="2")
        self.assertEqual(self.m.dedupe(), 0)
        c = self.m.write("我们一起去码头看了很久的海浪和夕阳啊", region="episodic", speaker="1")
        self.assertEqual(self.m.dedupe(), 1)
        self.assertIn(b["id"], self.m.nodes)
        self.assertTrue((a["id"] in self.m.nodes) ^ (c["id"] in self.m.nodes))

    def test_current_speaker_recalled_first(self):
        a = self.m.write("小明说：我养了一只橘猫叫大黄", speaker="1", speaker_name="小明")
        b = self.m.write("小红说：我养了一只橘猫叫团子", speaker="2", speaker_name="小红")
        for n in (a, b):
            self.m.set_vector(n["id"], "hash512", hash_embed(n["text"]))
        q = "我养的猫叫什么"
        for uid, want in (("1", a["id"]), ("2", b["id"])):
            m = MemoryStore.from_dict(self.m.to_dict(), self.m.vectors, rng=random.Random(0))
            hits = m.retrieve(q, k=2, query_vec=hash_embed(q), backend="hash512", speaker=uid, temperature=0)
            self.assertEqual(hits[0].node["id"], want)

    def test_abstraction_per_speaker_keeps_privacy(self):
        for t in ("我家猫今天打翻了水杯", "我家猫昨天抓了一只虫", "我家猫跟着我去了码头"):
            self.m.write(t, region="episodic", tags=["猫", "user:1"], speaker="1", speaker_name="小明")
        for t in ("群里的猫图好可爱", "猫猫表情包又更新了", "有人在群里晒猫"):
            self.m.write(t, region="episodic", tags=["猫", "user:2", "group:g"], speaker="2", speaker_name="小红")
        plans = self.m.abstraction_plans()
        self.assertEqual(len(plans), 2)
        self.m.consolidate()
        abstracts = [n for n in self.m.nodes.values() if n["region"] == "semantic"]
        by_speaker = {n["speaker"]: n for n in abstracts}
        self.assertEqual(set(by_speaker), {"1", "2"})
        self.assertIn("user:1", by_speaker["1"]["tags"])
        self.assertFalse(any(t.startswith("group:") for t in by_speaker["1"]["tags"]))  # 私聊来源的概括不进群
        self.assertIn("group:g", by_speaker["2"]["tags"])
        self.assertIn("小明", by_speaker["1"]["text"])

    def test_old_memories_migrated(self):
        old = {
            "nodes": {
                "m1": {"id": "m1", "text": "小明说：我喜欢草莓", "region": "episodic", "tier": "normal",
                       "tags": ["user:7"], "source": "other", "created": 1.0, "last_access": 1.0, "w0": 1.0},
                "m2": {"id": "m2", "text": "在书房读了一会儿书", "region": "episodic", "tier": "normal",
                       "tags": ["自主"], "source": "self", "created": 1.0, "last_access": 1.0, "w0": 1.0},
            },
            "counter": 2,
        }
        m = MemoryStore.from_dict(old)
        self.assertEqual(m.nodes["m1"]["speaker"], "7")
        self.assertEqual(m.nodes["m2"]["speaker"], "self")
        self.assertEqual(m.maintain(), {})

class TestMemoryBank(unittest.TestCase):
    """每人一个小记忆库 + 大记忆路由。"""

    def bank(self):
        b = MemoryBank(rng=random.Random(0))
        b.write("1", "我下周要考试", name="想睡觉", source="owner", tags=["user:1"])
        b.write("2", "我下周要搬家", name="小红", tags=["user:2"])
        b.write("2", "我在群里说过我养了只橘猫", name="小红", tags=["user:2", "group:g"])
        b.write(SELF, "在书房读了一会儿书", source="self")
        return b

    def test_route(self):
        b = self.bank()
        aliases = {"1": ["想睡觉", "主人"], "2": ["小红"]}
        self.assertEqual(b.route("你好呀", "1", aliases), {"1": 1.0, SELF: 0.9})
        self.assertEqual(b.route("小红最近在忙什么", "1", aliases), {"1": 1.0, SELF: 0.9, "2": 1.0})
        self.assertEqual(b.route("主人最近怎么样", "2", aliases), {"2": 1.0, SELF: 0.9, "1": 1.0})
        self.assertEqual(b.route("大家最近都在干嘛", "9", aliases), {SELF: 0.9, "1": 0.8, "2": 0.8})

    def test_joint_retrieval_and_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            a.bank = self.bank()
            a.social.touch("2", "小红")
            a.social.touch("3", "路人")

            def recall(uid, q, gid=None):
                w = a.bank.route(q, uid, a._aliases())
                hits = a.bank.retrieve(q, w, k=6, visible=lambda k, n: a._visible(k, n, uid, gid), update=False)
                return {(h.key, h.node["text"]) for h in hits}

            # 主人私聊问小红：能联合调用小红的库
            self.assertIn(("2", "我下周要搬家"), recall("1", "小红下周要干什么"))
            # 路人私聊问小红：只听得到小红在群里公开说过的
            got = recall("3", "小红下周要干什么，她养了什么")
            self.assertNotIn(("2", "我下周要搬家"), got)
            self.assertIn(("2", "我在群里说过我养了只橘猫"), got)
            # 群 g 里：只用在群 g 说过的；别的群里一条都不用
            self.assertNotIn(("2", "我下周要搬家"), recall("3", "小红下周要干什么", "g"))
            self.assertFalse({k for k, _ in recall("3", "小红养了什么", "h")} & {"2"})

    def test_prompt_groups_by_person_with_rough_time(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            a.bank = self.bank()
            a.social.touch("2", "小红")
            hits = [BankHit(k, Hit(n, 1.0), 1.0) for k, st in a.bank.stores.items() for n in st.nodes.values()]
            to_owner = a.memory_block(hits, "1")
            self.assertTrue(to_owner.startswith("〔主人（就是对方）说过的〕\n- 刚才：我下周要考试"))
            self.assertIn("〔小红说过的（不是对方）〕\n- 刚才：我下周要搬家", to_owner)
            self.assertTrue(to_owner.rstrip().endswith("〔你自己的经历〕\n- 刚才：在书房读了一会儿书"))
            to_red = a.memory_block(hits, "2")
            self.assertTrue(to_red.startswith("〔对方说过的〕"))
            self.assertIn("〔主人说过的（不是对方）〕", to_red)
            self.assertNotIn("想睡觉", to_red + to_owner)  # 主人一律称「主人」
            self.assertNotRegex(to_red + to_owner, r"\(\d+\)|\b[12]\b")  # 不把 QQ 号交给模型

    def test_time_label(self):
        tz = timezone.utc
        now_dt = datetime(2026, 10, 26, 20, 0, tzinfo=tz)
        ts = lambda **kw: (now_dt - timedelta(**kw)).timestamp()  # noqa: E731
        cases = [
            (ts(minutes=5), "刚才"), (ts(hours=5), "今天"), (ts(days=1), "昨天"), (ts(days=2), "前天"),
            (ts(days=4), "4天前"), (ts(days=9), "上周"), (ts(days=20), "2周前"), (ts(days=70), "2个月前"),
            (ts(days=800), "2年前"),
        ]
        for t, want in cases:
            self.assertEqual(time_label(t, now_dt, tz), want)

    def test_global_capacity(self):
        b = MemoryBank(capacity=3, rng=random.Random(0))
        for i, key in enumerate(["1", "2", "1", "3", "2"]):
            b.write(key, f"第{i}条完全不同的记忆{'甲乙丙丁戊'[i] * 4}")
        self.assertEqual(b.total(), 3)
        self.assertEqual(sum(len(st.cold) for st in b.stores.values()), 2)

    def test_consolidate_per_person(self):
        b = MemoryBank(rng=random.Random(0))
        for t in ("我家猫今天打翻了水杯", "我家猫昨天抓了一只虫", "我家猫跟着我去了码头"):
            b.write("1", t, name="小明", region="episodic", tags=["猫", "user:1"])
        b.write("2", "我也喜欢猫", name="小红", region="episodic", tags=["猫", "user:2"])
        report = b.consolidate()
        self.assertEqual(report["abstract"], 1)
        self.assertTrue(any(n["region"] == "semantic" for n in b.stores["1"].nodes.values()))
        self.assertFalse(any(n["region"] == "semantic" for n in b.stores["2"].nodes.values()))

    def test_save_load_and_legacy_migration(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            old = MemoryStore(rng=random.Random(0))
            a = old.write("我喜欢草莓", tags=["user:1"], speaker="1", speaker_name="小明")
            b = old.write("我喜欢蓝莓", tags=["user:2"], speaker="2", speaker_name="小红")
            c = old.write("在书房读了一会儿书", source="self")
            old.set_vector(a["id"], "hash512", hash_embed(a["text"]))
            (d / "memory.json").write_text(json.dumps(old.to_dict(), ensure_ascii=False), encoding="utf-8")
            (d / "vectors.json").write_text(json.dumps(old.vectors), encoding="utf-8")

            ag = Agent(Settings(data_dir=d), random.Random(0))
            ag.load()
            self.assertEqual({k: set(st.nodes) for k, st in ag.bank.stores.items()},
                             {"1": {a["id"]}, "2": {b["id"]}, SELF: {c["id"]}})
            self.assertIn(a["id"], ag.bank.stores["1"].vectors)
            self.assertEqual(ag.bank.name("2"), "小红")
            self.assertFalse(any(ag.bank.stores["1"].edges.get(a["id"], {})))  # 跨库的时间边不保留
            self.assertFalse((d / "memory.json").exists())
            self.assertTrue((d / "memory.json.bak-v0.4").exists())
            new = ag.bank.write("1", "我还喜欢西瓜")
            self.assertNotIn(new["id"], {a["id"], b["id"], c["id"]})  # 新 id 不和迁移来的重复

            ag.save()
            again = Agent(Settings(data_dir=d), random.Random(0))
            again.load()
            self.assertEqual({k: set(st.nodes) for k, st in again.bank.stores.items()},
                             {k: set(st.nodes) for k, st in ag.bank.stores.items()})

    def test_on_message_goes_to_speaker_store(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))

            async def run():
                await a.on_message(Incoming("1", "想睡觉", "我最喜欢的水果是草莓"))
                await a.on_message(Incoming("2", "小红", "我最喜欢的水果是草莓", group_id="g", is_tome=True))
                await a.job_ingest()

            asyncio.run(run())
            self.assertEqual([n["text"] for n in a.bank.stores["1"].nodes.values()], ["我最喜欢的水果是草莓"])
            red = list(a.bank.stores["2"].nodes.values())
            self.assertEqual(len(red), 1)
            self.assertIn("group:g", red[0]["tags"])
            self.assertEqual(a.bank.name("2"), "小红")
            text = a.memory_text("草莓", "1")
            self.assertIn("主人(1)·m", text)
            self.assertIn("刚才：我最喜欢的水果是草莓", text)


class TestPromptCache(unittest.TestCase):
    """提示词按「不变在前、变的在后」排列，历史分批截断，按需检索。"""

    def agent(self, d, with_llm=True):
        a = Agent(Settings(data_dir=Path(d), owners={"1"}, persona="人设" * 50), random.Random(0))
        a.weather["海"] = {"temp": 24, "humidity": 70, "wind": 8, "cloud": 0.3, "precipitation": 0}
        self.calls = []

        async def backend(messages, max_tokens, temperature):
            self.calls.append(messages)
            if not messages[0]["content"].startswith("你是"):
                return "前情：聊了工作和水果", 0, 0
            return f"回复{len(self.calls)}", 0, 0

        if with_llm:
            a.llm.backend = backend
        return a

    def chat_calls(self):
        return [m for m in self.calls if m[0]["content"].startswith("你是")]

    def test_system_prompt_is_stable(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d)
            msg = Incoming("1", "想睡觉", "你好")
            first = a._system_prompt("1", msg, "p:1")
            a.drives.bump("play", 50)
            a.econ.earn(30, "测试")
            a.activity["name"] = "读书"
            self.assertEqual(a._system_prompt("1", msg, "p:1"), first)
            self.assertNotIn("〈此刻〉", first.split("【说话方式】")[0])
            self.assertNotRegex(first, r"\d\d:\d\d")

    def test_prefix_grows_append_only_between_trims(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d)

            async def run():
                for i in range(14):
                    await a.on_message(Incoming("1", "想睡觉", f"第{i}件事：今天发生了一件很特别的事情编号{i}"))
                    await a.flush_background()

            asyncio.run(run())
            calls = self.chat_calls()
            breaks = 0
            for prev, cur in zip(calls, calls[1:]):
                if cur[: len(prev) - 1] != prev[:-1]:  # 上一轮除最后一条外，必须原样是这一轮的前缀
                    breaks += 1
            self.assertEqual(breaks, 1)  # 只有压缩那一次换了前缀
            self.assertTrue(a.summaries["p:1"].startswith("前情"))
            self.assertLessEqual(len(a.history["p:1"]), HIST_MAX)
            self.assertIn("【前情提要】前情", calls[-1][0]["content"])
            # 历史里只存原话，不存〈此刻〉
            self.assertFalse(any("〈此刻〉" in m["content"] for m in a.history["p:1"]))
            self.assertIn("〈此刻〉", calls[-1][-1]["content"])

    def test_trim_without_llm_uses_template(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d, with_llm=False)

            async def run():
                for i in range(11):
                    await a.on_message(Incoming("1", "想睡觉", f"第{i}件很特别的事情"))

            asyncio.run(run())
            self.assertEqual(len(a.history["p:1"]), HIST_KEEP)  # 第 11 轮到 22 条，超过 20，一次截到 8 条
            self.assertTrue(a.summaries["p:1"].startswith("之前聊到：第0件"))

    def test_small_talk_skips_recall_and_context(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d)
            a.bank.write("1", "我最喜欢的水果是草莓", tags=["user:1"])

            async def run():
                await a.on_message(Incoming("1", "想睡觉", "哈哈哈"))
                await a.on_message(Incoming("1", "想睡觉", "👌"))
                await a.on_message(Incoming("1", "想睡觉", "你还记得我喜欢什么水果吗"))

            asyncio.run(run())
            calls = self.chat_calls()
            self.assertEqual(calls[0][-1]["content"], "哈哈哈")
            self.assertEqual(calls[1][-1]["content"], "👌")
            self.assertIn("想起的事", calls[2][-1]["content"])
            self.assertIn("草莓", calls[2][-1]["content"])

    def test_recall_drops_what_was_just_said_and_weak_hits(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d)
            a.bank.write("1", "我下周三要去考驾照科目二", tags=["user:1"])
            weak = a.bank.write("1", "楼下便利店的关东煮还挺好吃", tags=["user:1"])
            hist = [{"role": "user", "content": "我下周三要去考驾照科目二"}]

            hits = asyncio.run(a._recall(Incoming("1", "x", "考驾照要注意什么"), "1", hist))
            self.assertEqual(hits, [])  # 刚说过的不重复放
            hits = asyncio.run(a._recall(Incoming("1", "x", "考驾照要注意什么"), "1", []))
            self.assertEqual([h.node["text"] for h in hits], ["我下周三要去考驾照科目二"])
            self.assertEqual(weak["hits"], 0)  # 没用上的不做检索强化

    def test_group_prompt_does_not_change_with_speaker(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d)

            async def run():
                await a.on_message(Incoming("2", "小红", "大肥鱼你家猫叫什么", group_id="g", is_tome=True))
                await a.on_message(Incoming("3", "阿杰", "大肥鱼今天天气怎么样", group_id="g", is_tome=True))

            asyncio.run(run())
            c1, c2 = self.chat_calls()
            self.assertEqual(c1[0], c2[0])  # system 相同
            self.assertNotIn("小红", c1[0]["content"])
            self.assertIn("说话的是阿杰", c2[-1]["content"])
            self.assertEqual(c2[1]["content"], "小红：大肥鱼你家猫叫什么")

    def test_history_and_summary_persist(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.agent(d, with_llm=False)
            asyncio.run(a.on_message(Incoming("1", "想睡觉", "今天去了海边")))
            a.summaries["p:1"] = "之前聊到：工作"
            a.save()
            b = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            b.load()
            self.assertEqual(b.history, a.history)
            self.assertEqual(b.summaries, {"p:1": "之前聊到：工作"})


class TestCards(unittest.TestCase):
    """印象卡：模型每晚整理，事实带有效期，常驻提示词。"""

    def test_apply_ops_keeps_history(self):
        from core import cards

        c = cards.empty_card()
        cards.apply_ops(c, [{"op": "add", "text": "喜欢蓝莓"}, {"op": "add", "text": "生日三月五号"}], 1.0)
        self.assertEqual([f["text"] for f in cards.current_facts(c)], ["喜欢蓝莓", "生日三月五号"])
        done = cards.apply_ops(c, [{"op": "update", "id": 1, "text": "喜欢草莓"}, {"op": "remove", "id": 2},
                                   {"op": "update", "id": 9, "text": "越界"}, "垃圾", {"op": "add", "text": "喜欢草莓"}], 2.0)
        self.assertEqual(done, {"add": 0, "update": 1, "remove": 1})  # 越界、重复、垃圾都忽略
        self.assertEqual([f["text"] for f in cards.current_facts(c)], ["喜欢草莓"])
        self.assertEqual({f["text"] for f in cards.past_facts(c)}, {"喜欢蓝莓", "生日三月五号"})
        self.assertIn("（以前：", cards.render(c))

    def test_caps_and_parse(self):
        from core import cards

        c = cards.empty_card()
        cards.apply_ops(c, [{"op": "add", "text": f"事实{i}"} for i in range(20)], 1.0)
        self.assertEqual(len(cards.current_facts(c)), cards.FACTS_CURRENT_MAX)
        self.assertLessEqual(len(cards.render(c)), 160)
        self.assertEqual(cards.parse_json('好的：\n```json\n{"profile": "p", "ops": []}\n```'), {"profile": "p", "ops": []})
        self.assertIsNone(cards.parse_json("不是 JSON"))
        self.assertIsNone(cards.parse_json('{"profile": '))

    def make(self, d, card_reply='{"profile": "主人是上班族，最近在准备驾照考试", "ops": [{"op": "add", "text": "喜欢草莓"}]}'):
        a = Agent(Settings(data_dir=Path(d), owners={"1"}, card_min_new=3), random.Random(0))
        a.weather["海"] = {"temp": 24, "humidity": 70, "wind": 8, "cloud": 0.3, "precipitation": 0}
        self.calls = {"chat": [], "card": []}

        async def chat(messages, max_tokens, temperature):
            self.calls["chat"].append(messages)
            return "嗯嗯", 0, 0

        async def card(messages, max_tokens, temperature):
            self.calls["card"].append(messages)
            if "最近的生活" in messages[0]["content"]:
                return "你最近在阳台种花，常和小灰玩。", 0, 0
            return card_reply, 0, 0

        a.llm.backend = chat
        a.card_backend = card
        return a

    def test_nightly_update_respects_threshold_and_uses_card_model(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make(d)
            for t in ("我最喜欢的水果是草莓", "下周三考驾照", "今天加班到九点"):
                a.bank.write("1", t, tags=["user:1"])
            a.bank.write("2", "我喜欢猫", name="小红", tags=["user:2"])
            for t in ("在书房读书", "给菜地浇水", "和小灰玩"):
                a.bank.write(SELF, t, source="self")
            r = asyncio.run(a.job_cards())
            self.assertEqual(sorted(r["updated"]), ["1", "2"])  # 第一次整理只要 1 条新记忆
            self.assertTrue(r["self"])
            self.assertEqual(self.calls["chat"], [])  # 全走整理卡片的模型
            self.assertIn("喜欢草莓", a.card_line("1"))
            self.assertIn("阳台种花", a.card_line(SELF))
            # 第二晚：主人只有 1 条新记忆，低于门槛 3，跳过
            a.bank.write("1", "今天下雨了", tags=["user:1"])
            r = asyncio.run(a.job_cards())
            self.assertEqual(r["updated"], [])
            # 手动 /整理卡片 不看门槛
            r = asyncio.run(a.update_cards(keys=["1"], force=True))
            self.assertEqual(r["updated"], ["1"])
            self.assertIn("今天下雨了", self.calls["card"][-1][1]["content"])

    def test_max_people_and_bad_json(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make(d, card_reply="我不会输出 JSON")
            a.s.card_max_people = 1
            for uid in ("2", "3"):
                a.bank.write(uid, f"群友{uid}说了些话", tags=[f"user:{uid}"])
            a.bank.write("3", "群友3又说了些话", tags=["user:3"])
            r = asyncio.run(a.update_cards())
            self.assertEqual(r["failed"], ["3"])  # 新记忆多的优先，坏 JSON 记为失败，卡片不变
            self.assertEqual(r["skipped"], 1)
            self.assertIsNone(a.card("3"))

    def test_cards_in_prompt_and_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make(d)
            a.social.touch("2", "小红")
            a.bank.write("1", "我最喜欢的水果是草莓", tags=["user:1"])
            a.bank.write("2", "我最近失恋了", name="小红", tags=["user:2"])
            a.bank.meta["2"]["card"] = {"profile": "小红最近失恋了", "facts": [], "updated": 1.0}
            for t in ("在书房读书", "给菜地浇水", "和小灰玩"):
                a.bank.write(SELF, t, source="self")
            asyncio.run(a.job_cards())

            async def say(uid, nick, text, gid=None):
                await a.on_message(Incoming(uid, nick, text, group_id=gid, is_tome=True))
                return self.calls["chat"][-1]

            owner = asyncio.run(say("1", "想睡觉", "今天好累"))
            self.assertIn("【你对主人的印象】主人是上班族", owner[0]["content"])
            self.assertIn("【你最近的生活】你最近在阳台种花", owner[0]["content"])
            self.assertNotIn("失恋", owner[-1]["content"])
            asked = asyncio.run(say("1", "想睡觉", "小红最近怎么样"))
            self.assertIn("你对小红的印象：小红最近失恋了", asked[-1]["content"])  # 主人私聊问起可以看
            group = asyncio.run(say("3", "阿杰", "小红最近怎么样", gid="g"))
            self.assertNotIn("失恋", group[0]["content"] + group[-1]["content"])  # 群里不外传
            red_group = asyncio.run(say("2", "小红", "大肥鱼你好呀", gid="g"))
            self.assertIn("你对TA的印象：小红最近失恋了", red_group[-1]["content"])
            self.assertNotIn("失恋", red_group[0]["content"])  # 群聊 system 不随说话人变
            # 卡片持久化
            a.save()
            b = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            b.load()
            self.assertEqual(b.card_line("1"), a.card_line("1"))
            self.assertEqual(b.card_line(SELF), a.card_line(SELF))
            self.assertIn("对主人的印象", b.card_text("1"))


class TestHostHooks(unittest.TestCase):
    def test_llm_backend_and_cost(self):
        from core.llm import LLM

        seen = {}

        async def backend(messages, max_tokens, temperature):
            seen["messages"] = messages
            return " 好呀 ", 1000, 500

        llm = LLM("", "", "", price_in=0.002, price_out=0.008, backend=backend)
        self.assertTrue(llm.enabled)
        out = asyncio.run(llm.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]))
        self.assertEqual(out, "好呀")
        self.assertAlmostEqual(llm.daily_burn(), 0.002 + 0.004)

    def test_cache_hit_pricing(self):
        from core.llm import LLM

        async def backend(messages, max_tokens, temperature):
            return "好", 1000, 500, 800  # 1000 输入里 800 命中缓存

        llm = LLM("", "", "", price_in=0.002, price_out=0.008, backend=backend, price_in_hit=0.0005)
        asyncio.run(llm.chat([{"role": "user", "content": "u"}]))
        # 200 未命中 × 0.002 + 800 命中 × 0.0005 + 500 输出 × 0.008（每千 token）
        self.assertAlmostEqual(llm.daily_burn(), 0.0004 + 0.0004 + 0.004)
        report = llm.report()
        self.assertIn("命中缓存 800", report)
        self.assertIn("缓存命中率：今日 80%", report)
        self.assertIn("省下约 ¥0.0012", report)  # 800 × (0.002 − 0.0005) / 1000

    def test_usage_parsing(self):
        from core.llm import usage_from_openai

        self.assertEqual(usage_from_openai({"prompt_tokens": 100, "completion_tokens": 5, "prompt_cache_hit_tokens": 80}), (100, 5, 80))
        self.assertEqual(
            usage_from_openai({"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 64}}),
            (100, 5, 64),
        )
        self.assertEqual(usage_from_openai({"prompt_tokens": 10, "completion_tokens": 1}), (10, 1, 0))
        self.assertEqual(usage_from_openai(None), (0, 0, 0))

    def test_status_line_is_a_short_phrase(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d)), random.Random(0))
            a.weather["海"] = {"temp": 24, "humidity": 70, "wind": 8, "cloud": 0.3, "precipitation": 0}
            calm = a.status_line("你好")
            self.assertNotIn("驱力", calm)
            self.assertNotIn("湿度", calm)
            self.assertNotIn("金币", calm)
            self.assertNotIn("小灰", calm)
            self.assertLess(len(calm), 30)
            a.drives.value.update(rest=90, social=80, play=75)
            a.body["hunger"] = 80
            busy = a.status_line("你好")
            self.assertIn("有点困，有点想找人说话，有点饿", busy)  # 最多两个驱力，按强弱
            self.assertNotIn("想出去玩", busy)
            self.assertIn("小灰在", a.status_line("小灰今天乖吗"))
            self.assertIn("金币", a.status_line("你钱够花吗"))

    def test_embed_provider_fallback(self):
        from core.embedding import HASH_BACKEND, Embedder

        async def broken(texts):
            raise RuntimeError("down")

        e = Embedder(provider=broken, provider_name="x")
        backend, vecs = asyncio.run(e.embed(["你好"]))
        self.assertEqual(backend, HASH_BACKEND)
        self.assertEqual(len(vecs[0]), 512)

    def test_group_persona(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), persona="私聊人设甲", group_persona="群聊人设乙"))
            private = a._system_prompt("1", Incoming("1", "x", "hi"), [])
            group = a._system_prompt("1", Incoming("1", "x", "hi", group_id="g"), [])
            self.assertIn("私聊人设甲", private)
            self.assertNotIn("群聊人设乙", private)
            self.assertIn("群聊人设乙", group)
            b = Agent(Settings(data_dir=Path(d), persona="私聊人设甲"))
            self.assertIn("私聊人设甲", b._system_prompt("1", Incoming("1", "x", "hi", group_id="g"), []))

    def test_timezone(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), tz="Asia/Shanghai"))
            self.assertEqual(a._dt().utcoffset().total_seconds(), 8 * 3600)
            b = Agent(Settings(data_dir=Path(d), tz="Not/AZone"))
            self.assertEqual(b._dt().utcoffset().total_seconds(), 8 * 3600)


class TestAgent(unittest.TestCase):
    def test_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), owners={"1"}, owner_nick="小鱼干"), random.Random(0))
            a.weather["海"] = {"temp": 24, "humidity": 70, "wind": 8, "cloud": 0.3, "precipitation": 0}

            async def run():
                r = await a.on_message(Incoming("1", "小鱼干", "我最喜欢的水果是草莓"))
                self.assertTrue(r)
                await a.job_ingest()
                a.eco.meals.append({"name": "白粥", "ts": a.last_tick})
                await a.on_message(Incoming("1", "小鱼干", "我吃完了"))
                self.assertFalse(a.eco.meals)
                for _ in range(12):
                    a.activity["until"] = 0
                    await a.job_tick()
                return await a.job_diary()

            diary = asyncio.run(run())
            self.assertTrue(diary)
            self.assertIn("草莓", a.memory_text("喜欢什么水果", "1"))
            self.assertEqual(a.job_selftest(), [])
            a.save()
            b = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            b.load()
            ids = lambda ag: {(k, n) for k, st in ag.bank.stores.items() for n in st.nodes}  # noqa: E731
            self.assertEqual(ids(b), ids(a))
            self.assertIn("1", b.bank.stores)


if __name__ == "__main__":
    unittest.main()
