# -*- coding: utf-8 -*-
"""OpenAI 兼容的大模型客户端（纯标准库实现，无需 pip 安装依赖）。

设计目标（对应《研究方案》2.4 节的可复现要求）：
1. 每次调用都记录服务端返回的真实 model 版本号 —— API 模型会静默升级，这是复现性命门；
2. 失败自动重试 + 指数退避，避免新手被 429/超时打断整个实验；
3. 结果落 sqlite 缓存，重跑不会重复花钱；
4. 零第三方依赖，只改 base_url / model 就能在 DeepSeek / 通义千问 / 智谱 / 火山方舟之间切换。

用法：
    from llm_client import LLMClient
    client = LLMClient()                      # 从环境变量读配置，见 .env.example
    r = client.chat([{"role": "user", "content": "1+1=?"}])
    print(r.text, r.model_version, r.prompt_tokens)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 4
DEFAULT_MAX_TOKENS = 8192
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ChatResult:
    """一次调用的完整记录。runs.csv 里的字段全部来自这里。"""

    text: str
    model: str
    model_version: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    from_cache: bool = False
    finish_reason: str = ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResponseCache:
    """极简 sqlite 缓存：key = 请求指纹，value = 完整响应。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class LLMClient:
    """真实 API 客户端。配置优先级：构造参数 > 环境变量 > 默认值。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache_path: str | Path = ".cache/llm_cache.sqlite",
        use_cache: bool = True,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or ""
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache = ResponseCache(cache_path) if use_cache else None
        if not self.api_key:
            raise RuntimeError(
                "没有找到 API Key。请复制 .env.example 为 .env 并填入 LLM_API_KEY，"
                "或者先用 python src/run_experiment.py --mock 跑通流程。"
            )

    # ------------------------------------------------------------------ core
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed

        cache_key = _sha256(
            json.dumps(
                {"url": self.base_url, "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return ChatResult(
                    text=hit["text"],
                    model=hit.get("model", self.model),
                    model_version=hit.get("model_version", self.model),
                    prompt_tokens=hit.get("prompt_tokens", 0),
                    completion_tokens=hit.get("completion_tokens", 0),
                    latency_ms=0,
                    from_cache=True,
                    finish_reason=hit.get("finish_reason", ""),
                )

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.time()
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                latency_ms = int((time.time() - started) * 1000)
                result = self._parse(raw, latency_ms)
                if self.cache is not None:
                    self.cache.put(
                        cache_key,
                        {
                            "text": result.text,
                            "model": result.model,
                            "model_version": result.model_version,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "finish_reason": result.finish_reason,
                        },
                    )
                return result
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:  # noqa: BLE001 - 读不到响应体就只看状态码
                    pass
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code not in RETRY_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(
            f"调用失败（已重试 {self.max_retries} 次）：{last_error}"
        ) from last_error

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _parse(raw: str, latency_ms: int) -> ChatResult:
        obj = json.loads(raw)
        choices = obj.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage = obj.get("usage") or {}
        finish_reason = (choices[0].get("finish_reason") or "") if choices else ""
        return ChatResult(
            text=text,
            model=obj.get("model") or "",
            model_version=obj.get("model") or "unknown",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )


class MockClient:
    """不联网的假客户端：验证代码结构用。

    它按 prompt 的哈希随机给一个字母，并不知道标准答案。
    所以：能证明"流程跑通"，但不能用来得出任何研究结论。
    要看"能否还原已知效应"，请用 run_experiment.py --mock（仿真模式，带真值）。
    """

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        joined = "\n".join(m.get("content", "") for m in messages)
        digest = _sha256(joined + f"|{temperature}|{seed}")
        letter = "ABCD"[int(digest[:8], 16) % 4]
        body = f"（mock）根据题意分析，可以排除干扰项。\n答案：{letter}"
        return ChatResult(
            text=body,
            model=self.model,
            model_version="mock-v1",
            prompt_tokens=max(1, len(joined) // 2),
            completion_tokens=len(body) // 2,
            latency_ms=0,
        )


def ping(client: LLMClient) -> None:
    """连通性自检：新同学第一天用这个确认 key 有没有配对。"""
    result = client.chat([{"role": "user", "content": "只回复两个字：你好"}], max_tokens=16)
    print(f"✅ 连通成功  版本={result.model_version}  回复={result.text!r}")
    print(f"   token: 输入 {result.prompt_tokens} / 输出 {result.completion_tokens}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API 连通性自检")
    parser.add_argument("--mock", action="store_true", help="用假客户端，不联网")
    args = parser.parse_args()

    if args.mock:
        print(MockClient().chat([{"role": "user", "content": "你好"}]).text)
    else:
        ping(LLMClient())
#b4681ef42be341d3a03dfa0d8ea746df.aTKIuwzSz0GuHlom