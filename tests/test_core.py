"""核心算法测试（不依赖 AstrBot）。运行：python -m unittest discover -s tests"""
import asyncio
import math
import random
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import gua, sun  # noqa: E402
from core.agent import Agent, Incoming, Settings  # noqa: E402
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

    def test_prompt_labels_who_said_it(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            a.social.touch("2", "小红")
            mine = a.memory.write("主人说：我下周要考试", speaker="1")
            hers = a.memory.write("小红说：我下周要搬家", speaker="2", speaker_name="小红")
            own = a.memory.write("在书房读了一会儿书", source="self")
            hits = [Hit(n, 1.0) for n in (mine, hers, own)]
            to_owner = a._system_prompt("1", Incoming("1", "主人", "hi"), hits)
            self.assertIn("【主人（就是对方）说的】主人说：我下周要考试", to_owner)
            self.assertIn("【小红说的（不是对方）】小红说：我下周要搬家", to_owner)
            self.assertIn("【你自己的经历】在书房读了一会儿书", to_owner)
            to_red = a._system_prompt("2", Incoming("2", "小红", "hi"), hits)
            self.assertIn("【对方本人说的】小红说：我下周要搬家", to_red)
            self.assertIn("【主人说的（不是对方）】", to_red)
            self.assertNotIn("1", a.speaker_label(mine, "2").replace("主人", ""))  # 不把 QQ 号交给模型

    def test_on_message_records_speaker(self):
        with tempfile.TemporaryDirectory() as d:
            a = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))

            async def run():
                await a.on_message(Incoming("1", "想睡觉", "我最喜欢的水果是草莓"))
                await a.on_message(Incoming("2", "小红", "我最喜欢的水果是草莓", group_id="g", is_tome=True))
                await a.job_ingest()

            asyncio.run(run())
            nodes = sorted(a.memory.nodes.values(), key=lambda n: n["speaker"])
            self.assertEqual([(n["speaker"], n["speaker_name"]) for n in nodes], [("1", "想睡觉"), ("2", "小红")])
            self.assertIn("来源：主人(1)", a.memory_text("草莓"))


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
            self.assertIn("草莓", a.memory_text("喜欢什么水果"))
            self.assertEqual(a.job_selftest(), [])
            a.save()
            b = Agent(Settings(data_dir=Path(d), owners={"1"}), random.Random(0))
            b.load()
            self.assertEqual(b.memory.nodes.keys(), a.memory.nodes.keys())


if __name__ == "__main__":
    unittest.main()
