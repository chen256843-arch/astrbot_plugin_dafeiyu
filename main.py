"""大肥鱼 AstrBot 插件入口：收发消息、命令、定时节律。核心逻辑都在 core/ 里。"""
from __future__ import annotations

import asyncio

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, ResultContentType, filter
from astrbot.api.star import Context, Star, StarTools

from .core import Agent, Incoming, Settings
from .core.agent import JOB_INTERVALS
from .core.economy import FURNITURE, GOODS
from .core.llm import today
from .core.util import now
from .core.world import ROOMS

PLUGIN_NAME = "astrbot_plugin_dafeiyu"
LOOP_SECONDS = 30
# 每日任务：(名称, 时, 分)
DAILY = (("consolidate", 3, 0), ("diary", 22, 30))
CHORES = {"浇水": "浇水", "收菜": "收菜", "种地": "种地", "打扫": "家务", "喂猫": "喂猫", "剪花": "剪花", "拾荒": "拾荒"}


class DaFeiYu(Star):
    """大肥鱼：记忆闭环 + 驱力决策 + 浮空岛环境 + 经济系统的长期陪伴角色。

    私聊直接说话；群聊 @ 她或聊到她感兴趣的话题。
    /状态 /岛 /房间 /农场 /日记 /商店 /关系
    主人：/记忆 [关键词] /记住 内容 /购买 物品 [数量] /浇水 /收菜 /种地 [作物] /打扫 /喂猫 /剪花 /拾荒
          /门 房间A 房间B open|closed|locked /禁忌 QQ号 话题 /自检 /成本 /余额 数值 /巩固
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.owners = {str(x).strip() for x in config.get("owners", []) if str(x).strip()}
        self.groups = {str(x).strip() for x in config.get("groups", []) if str(x).strip()}
        custom_llm = config.get("llm_source", "astrbot") == "custom"
        settings = Settings(
            data_dir=StarTools.get_data_dir(PLUGIN_NAME),
            name=config.get("name", "大肥鱼"),
            group_persona=config.get("group_persona", ""),
            aliases=list(config.get("aliases", [])),
            owners=self.owners,
            owner_nick=config.get("owner_nick", ""),
            interests=list(config.get("interests", [])) or ["猫", "书", "花", "种菜", "天气", "海"],
            home=config.get("home", "海"),
            group_p0=float(config.get("group_p0", 0.03)),
            proactive=bool(config.get("proactive", True)),
            quiet_start=int(config.get("quiet_start", 23)),
            quiet_end=int(config.get("quiet_end", 8)),
            embed_api_key=config.get("embed_api_key", ""),
            embed_base_url=config.get("embed_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            embed_model=config.get("embed_model", "text-embedding-v4"),
            llm_api_key=config.get("llm_api_key", "") if custom_llm else "",
            llm_base_url=config.get("llm_base_url", "https://api.deepseek.com"),
            llm_model=config.get("llm_model", "deepseek-chat"),
            price_in=float(config.get("price_in", 0.002)),
            price_out=float(config.get("price_out", 0.008)),
            hot_capacity=int(config.get("hot_capacity", 5000)),
            tz=config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai",
        )
        if config.get("persona"):
            settings.persona = config["persona"]
        self.agent = Agent(settings)
        if not custom_llm:
            self.agent.llm.backend = self._provider_chat
        if config.get("embedding_provider_id"):
            self.agent.embedder.provider = self._provider_embed
            self.agent.embedder.provider_name = config["embedding_provider_id"]
        self._task: asyncio.Task | None = None
        self._last_run: dict[str, float] = {}
        self._last_heal: dict[str, float] = {}

    # ================================================================ 生命周期
    async def initialize(self) -> None:
        self.agent.load()
        if not self.owners:
            logger.warning("大肥鱼：还没有在插件配置里填主人 QQ 号（owners）")
        logger.info(f"大肥鱼已载入：{self.agent.memory_stats_text()}")
        self._task = asyncio.create_task(self._loop())

    async def terminate(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.agent.save()

    # ================================================================ 模型接入
    async def _chat_provider(self):
        pid = self.config.get("llm_provider_id", "")
        prov = self.context.get_provider_by_id(pid) if pid else None
        if prov is None or not hasattr(prov, "text_chat"):
            prov = await self.context.get_using_provider_async()
        if prov is None:
            raise RuntimeError("AstrBot 里没有可用的对话模型提供商")
        return prov

    async def _provider_chat(self, messages: list[dict], max_tokens: int, temperature: float):
        """用 AstrBot 面板里配好的提供商（例如 DeepSeek）对话。"""
        prov = await self._chat_provider()
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        prompt = rest[-1]["content"] if rest and rest[-1]["role"] == "user" else None
        contexts = rest[:-1] if prompt is not None else rest
        resp = await prov.text_chat(prompt=prompt, contexts=contexts, system_prompt=system or None)
        usage = getattr(resp, "usage", None)
        return resp.completion_text, (usage.input if usage else 0), (usage.output if usage else 0)

    async def _provider_embed(self, texts: list[str]) -> list[list[float]]:
        prov = self.context.get_provider_by_id(self.config["embedding_provider_id"])
        if prov is None or not hasattr(prov, "get_embeddings"):
            raise RuntimeError("找不到嵌入模型提供商：" + self.config["embedding_provider_id"])
        return await prov.get_embeddings(texts)

    # ================================================================ 定时节律
    def _jobs(self) -> dict:
        a = self.agent
        return {
            "tick": self._job_tick,
            "weather": a.job_weather,
            "ingest": a.job_ingest,
            "maintain": self._job_maintain,
            "consolidate": self._job_consolidate,
            "diary": a.job_diary,
            "selftest": self._job_selftest,
            "save": self._job_save,
        }

    async def _job_tick(self) -> None:
        for uid, text in await self.agent.job_tick():
            await self._send_owner(uid, text)

    async def _job_maintain(self) -> None:
        fixed = self.agent.job_maintain()
        if fixed:
            logger.info(f"记忆图维护：{fixed}")

    async def _job_consolidate(self) -> None:
        logger.info(f"夜间巩固：{await self.agent.job_consolidate()}")

    async def _job_selftest(self) -> None:
        problems = self.agent.job_selftest()
        if problems:
            logger.warning(f"自测问题：{problems}")

    async def _job_save(self) -> None:
        self.agent.save()

    async def _run(self, name: str) -> None:
        try:
            await self._jobs()[name]()
        except Exception as e:  # noqa: BLE001
            logger.error(f"大肥鱼任务 {name} 失败: {e}", exc_info=True)

    async def _loop(self) -> None:
        """自带的轻量调度：间隔任务 + 每日定点任务 + 健康监测自愈。"""
        interval_jobs = ("tick", "weather", "ingest", "maintain", "selftest", "save")
        while True:
            try:
                t = now()
                for name in interval_jobs:
                    if t - self._last_run.get(name, 0) >= JOB_INTERVALS[name]:
                        self._last_run[name] = t
                        await self._run(name)
                local = self.agent._dt()
                done = self.agent.host.setdefault("daily", {})
                for name, h, m in DAILY:
                    if (local.hour, local.minute) >= (h, m) and done.get(name) != local.strftime("%Y-%m-%d"):
                        done[name] = local.strftime("%Y-%m-%d")
                        await self._run(name)
                await self._watchdog(t)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"大肥鱼调度循环出错: {e}", exc_info=True)
            await asyncio.sleep(LOOP_SECONDS)

    async def _watchdog(self, t: float) -> None:
        """最后成功时间 vs 期望间隔；滞后超过 3 倍则限流重跑。"""
        for job, lag, _ in self.agent.health_report():
            if job not in self.agent.health or lag <= 3 * JOB_INTERVALS[job]:
                continue
            if t - self._last_heal.get(job, 0) < 1800:
                continue
            self._last_heal[job] = t
            logger.warning(f"任务 {job} 已滞后 {lag / 60:.0f} 分钟，尝试自愈重跑")
            await self._run(job)

    async def _send_owner(self, uid: str, text: str) -> None:
        umo = self.agent.host.get("owner_umo", {}).get(uid)
        if not umo:
            logger.info(f"主人 {uid} 还没私聊过机器人，无法主动发消息")
            return
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"主动消息发送失败: {e}")

    # ================================================================ 对话
    def _is_command(self, raw: str) -> bool:
        prefixes = self.context.get_config().get("wake_prefix", ["/"]) or []
        return any(p and raw.startswith(p) for p in prefixes)

    def _owner(self, event: AstrMessageEvent) -> bool:
        return str(event.get_sender_id()) in self.owners

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def on_message(self, event: AstrMessageEvent):
        raw = (event.message_obj.message_str or "").strip()
        if not raw or self._is_command(raw):
            return
        uid = str(event.get_sender_id())
        if not uid or uid == str(event.get_self_id()):
            return
        gid = str(event.get_group_id() or "") or None
        if gid and self.groups and gid not in self.groups:
            return
        if gid is None and uid in self.owners:
            self.agent.host.setdefault("owner_umo", {})[uid] = event.unified_msg_origin
        text = (event.message_str or raw).strip()
        reply = await self.agent.on_message(
            Incoming(uid, event.get_sender_name(), text, gid, bool(event.is_at_or_wake_command) if gid else True)
        )
        if reply:
            result = event.plain_result(reply)
            mode = self.config.get("segment_reply", "private")
            if mode == "all" or (mode == "private" and gid is None):
                # 标记成模型结果，交给 AstrBot 的「分段回复」按句切开逐条发送
                result.set_result_content_type(ResultContentType.LLM_RESULT)
            yield result
            event.stop_event()  # 不再交给 AstrBot 默认的 LLM 回复

    # ================================================================ 查看类命令
    @staticmethod
    def _args(event: AstrMessageEvent) -> list[str]:
        return (event.message_str or "").split()[1:]

    @filter.command("状态")
    async def cmd_status(self, event: AstrMessageEvent):
        yield event.plain_result(self.agent.status_text())

    @filter.command("岛", alias={"地图"})
    async def cmd_map(self, event: AstrMessageEvent):
        yield event.plain_result(self.agent.world.render_map())

    @filter.command("房间")
    async def cmd_rooms(self, event: AstrMessageEvent):
        w = self.agent.world
        lines = [
            f"{r}：{s['temp']:.1f}°C 灰尘{s['dust']:.0f}" + (" ← 她" if w.her["room"] == r and w.indoors() else "")
            for r, s in w.rooms.items()
        ]
        lines.append("门：" + " ".join(f"{a}-{b}:{st}" for a, b, st in w.doors))
        yield event.plain_result("\n".join(lines))

    @filter.command("农场", alias={"菜地"})
    async def cmd_farm(self, event: AstrMessageEvent):
        yield event.plain_result(self.agent.farm_text())

    @filter.command("日记")
    async def cmd_diary(self, event: AstrMessageEvent):
        if not self.agent.diaries:
            yield event.plain_result("还没写过日记")
            return
        d = self.agent.diaries[-1]
        yield event.plain_result(f"{d['date']}\n{d['text']}")

    @filter.command("商店")
    async def cmd_shop(self, event: AstrMessageEvent):
        yield event.plain_result(f"💰 {self.agent.econ.coins} 金币\n{self.agent.econ.catalog()}")

    @filter.command("关系")
    async def cmd_relation(self, event: AstrMessageEvent):
        args = self._args(event)
        uid = args[0] if args else str(event.get_sender_id())
        if uid != str(event.get_sender_id()) and not self._owner(event):
            yield event.plain_result("只能看自己的关系哦")
            return
        r = self.agent.social.rel.get(uid)
        if not r:
            yield event.plain_result("还不认识呢")
            return
        text = f"{r['nick'] or uid}：{self.agent.social.tier(uid)} 亲密{r['intimacy']:.0f} 信任{r['trust']:.0f} 消息{r['msgs']}"
        if r["taboos"]:
            text += f"\n禁忌：{'、'.join(r['taboos'])}"
        yield event.plain_result(text)

    # ================================================================ 主人命令
    @filter.command("记忆")
    async def cmd_memory(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        q = " ".join(self._args(event))
        uid = str(event.get_sender_id())
        yield event.plain_result(self.agent.memory_text(q, uid) if q else self.agent.memory_stats_text())

    @filter.command("记住")
    async def cmd_remember(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        text = " ".join(self._args(event))
        yield event.plain_result(self.agent.remember(text, str(event.get_sender_id())) if text else "要记住什么？")

    @filter.command("购买", alias={"买"})
    async def cmd_buy(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        args = self._args(event)
        if not args:
            yield event.plain_result("买什么？发 /商店 看看")
            return
        name = args[0]
        qty = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1
        a = self.agent
        if name in FURNITURE:
            msg = a.econ.buy_furniture(name)
        elif name in GOODS and name.endswith("种子"):
            msg = a.chore("种地", name[:-2])
        elif name in GOODS:
            msg = a.econ.buy_goods(name, qty, a.eco.pantry)
        else:
            msg = f"商店没有「{name}」"
        yield event.plain_result(msg)

    async def _chore(self, event: AstrMessageEvent, name: str):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        args = self._args(event)
        text = self.agent.chore(CHORES[name], args[0] if args else None)
        self.agent._event(text)
        yield event.plain_result(text)

    @filter.command("浇水")
    async def cmd_water(self, event: AstrMessageEvent):
        async for r in self._chore(event, "浇水"):
            yield r

    @filter.command("收菜")
    async def cmd_harvest(self, event: AstrMessageEvent):
        async for r in self._chore(event, "收菜"):
            yield r

    @filter.command("种地")
    async def cmd_plant(self, event: AstrMessageEvent):
        async for r in self._chore(event, "种地"):
            yield r

    @filter.command("打扫")
    async def cmd_clean(self, event: AstrMessageEvent):
        async for r in self._chore(event, "打扫"):
            yield r

    @filter.command("喂猫")
    async def cmd_feed_cat(self, event: AstrMessageEvent):
        async for r in self._chore(event, "喂猫"):
            yield r

    @filter.command("剪花")
    async def cmd_cut_flower(self, event: AstrMessageEvent):
        async for r in self._chore(event, "剪花"):
            yield r

    @filter.command("拾荒")
    async def cmd_scavenge(self, event: AstrMessageEvent):
        async for r in self._chore(event, "拾荒"):
            yield r

    @filter.command("门")
    async def cmd_door(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        args = self._args(event)
        if len(args) != 3 or args[2] not in ("open", "closed", "locked"):
            yield event.plain_result("用法：/门 房间A 房间B open|closed|locked，房间：" + " ".join([*ROOMS, "户外"]))
            return
        ok = self.agent.world.set_door(*args)
        yield event.plain_result("好了" if ok else "这两个房间之间没有门")

    @filter.command("禁忌")
    async def cmd_taboo(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        args = self._args(event)
        if len(args) < 2:
            yield event.plain_result("用法：/禁忌 QQ号 话题")
            return
        self.agent.social.add_taboo(args[0], " ".join(args[1:]))
        yield event.plain_result("记下了，不会跟 TA 聊这个")

    @filter.command("自检")
    async def cmd_selftest(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        problems = self.agent.job_selftest()
        lines = ["自测：" + ("全部通过" if not problems else "；".join(problems))]
        for job, lag, ok in self.agent.health_report():
            never = lag > 1e8
            icon = "⏳" if never else "✅" if ok else "⚠️"
            lag_txt = "还没到时间" if never else f"{lag / 60:.0f} 分钟前"
            lines.append(f"{icon} {job}：{lag_txt}（期望每 {JOB_INTERVALS[job] / 60:.0f} 分钟）")
        lines.append(self.agent.memory_stats_text())
        lines.append(f"主动消息：{'已记录主人会话' if self.agent.host.get('owner_umo') else '主人还没私聊过，暂时发不了'}")
        yield event.plain_result("\n".join(lines))

    @filter.command("成本")
    async def cmd_cost(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        yield event.plain_result(self.agent.llm.report())

    @filter.command("余额")
    async def cmd_balance(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        args = self._args(event)
        try:
            v = float(args[0])
        except (IndexError, ValueError):
            yield event.plain_result("用法：/余额 123.45（DeepSeek 账户余额，用来估算还能用几天）")
            return
        self.agent.llm.balance = {"value": v, "ts": now(), "day": today()}
        yield event.plain_result(self.agent.llm.report())

    @filter.command("巩固")
    async def cmd_consolidate(self, event: AstrMessageEvent):
        if not self._owner(event):
            yield event.plain_result("这个只有主人能用哦")
            return
        r = await self.agent.job_consolidate()
        yield event.plain_result(f"重放{r['replay']} 合并{r['merge']} 抽象{r['abstract']} 剪边{r['prune']} 下沉{r['forget']}")

