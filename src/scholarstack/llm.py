"""LLM access through LangChain, against any OpenAI-compatible endpoint (Groq by default).

* SQLite response cache -> experiments are resumable and re-runs cost nothing.
* Rate limiter + automatic retries + fallback model -> survives free-tier limits.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from .config import CFG, env, path


class SQLiteCache(BaseCache):
    """Minimal persistent cache keyed by (prompt, model settings)."""

    def __init__(self, db_file):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_file), check_same_thread=False)
        self._db.execute("create table if not exists cache (key text primary key, value text)")

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256((prompt + "\x1f" + llm_string).encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str):
        with self._lock:
            row = self._db.execute("select value from cache where key = ?", (self._key(prompt, llm_string),)).fetchone()
        if not row:
            return None
        from langchain_core.load import loads

        return [loads(g) for g in json.loads(row[0])]

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        from langchain_core.load import dumps

        with self._lock:
            self._db.execute("insert or replace into cache values (?, ?)",
                             (self._key(prompt, llm_string), json.dumps([dumps(g) for g in return_val])))
            self._db.commit()

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self._db.execute("delete from cache")
            self._db.commit()


_CACHE_SET = False
_LIMITER = InMemoryRateLimiter(requests_per_second=0.4, check_every_n_seconds=0.2, max_bucket_size=2)


def chat_model(model: str | None = None, json_mode: bool = True, max_tokens: int | None = None,
               with_fallback: bool = True):
    global _CACHE_SET
    if not _CACHE_SET:
        set_llm_cache(SQLiteCache(path("cache") / "llm_cache.sqlite"))
        _CACHE_SET = True
    kwargs = dict(base_url=env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                  api_key=env("GROQ_API_KEY", required=True),
                  temperature=CFG["llm"]["temperature"], max_tokens=max_tokens or CFG["llm"]["max_tokens"],
                  max_retries=6, timeout=120, rate_limiter=_LIMITER)
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    primary = ChatOpenAI(model=model or env("LLM_MODEL", "llama-3.3-70b-versatile"), **kwargs)
    fallback_name = env("LLM_FALLBACK_MODEL")
    if with_fallback and fallback_name and fallback_name != (model or env("LLM_MODEL")):
        return primary.with_fallbacks([ChatOpenAI(model=fallback_name, **kwargs)])
    return primary


def model_name(model: str | None = None) -> str:
    return model or env("LLM_MODEL", "llama-3.3-70b-versatile")
