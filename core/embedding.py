"""语义向量：优先调用 OpenAI 兼容 /embeddings（默认 DashScope text-embedding-v4，512 维），
无密钥或调用失败时退化为本地特征哈希向量。两种后端的向量不可互比，按 backend 字段区分。"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Awaitable, Callable

from .util import tokens

log = logging.getLogger("dafeiyu.embedding")

DIM = 512
HASH_BACKEND = "hash512"


def hash_embed(text: str, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    for t in tokens(text):
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign * (1.5 if len(t) > 1 else 1.0)
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class Embedder:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "text-embedding-v4",
        dim: int = DIM,
        provider: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        provider_name: str = "",
    ):
        """provider：宿主框架注入的向量接口（如 AstrBot 的嵌入提供商），优先于 api_key。"""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.provider = provider
        self.provider_name = provider_name

    @property
    def backend(self) -> str:
        if self.provider:
            return f"provider:{self.provider_name}"
        return f"{self.model}:{self.dim}" if self.api_key else HASH_BACKEND

    async def embed(self, texts: list[str]) -> tuple[str, list[list[float]]]:
        """返回 (backend, vectors)。远端失败时整批退化为哈希向量。"""
        if not texts:
            return self.backend, []
        if self.provider:
            try:
                return self.backend, await self.provider(texts)
            except Exception as e:  # noqa: BLE001
                log.warning("嵌入提供商调用失败，退化为哈希向量: %s", e)
        elif self.api_key:
            try:
                return self.backend, await self._remote(texts)
            except Exception as e:  # noqa: BLE001
                log.warning("embedding 调用失败，退化为哈希向量: %s", e)
        return HASH_BACKEND, [hash_embed(t, DIM) for t in texts]

    async def _remote(self, texts: list[str]) -> list[list[float]]:
        import httpx

        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(0, len(texts), 10):  # DashScope 单批上限 10
                batch = texts[i : i + 10]
                r = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch, "dimensions": self.dim, "encoding_format": "float"},
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                out.extend(d["embedding"] for d in data)
        return out
