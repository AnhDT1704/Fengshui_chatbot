"""
gemini.py – Central factory for Gemini LLM instances with API-key fallback.

Reads `GOOGLE_API_KEY1`, `GOOGLE_API_KEY2`, ... `GOOGLE_API_KEYN` from .env
(also accepts unnumbered `GOOGLE_API_KEY` as the final fallback).

Builds a chain where the primary LLM uses key #1; if it raises (quota
exhausted, transient 5xx, etc.) LangChain's `.with_fallbacks()` automatically
tries the next key, and so on.

Two public helpers:
    make_llm(temperature, max_tokens)     – plain chat model (used by
                                            supervisor + small_talk)
    make_llm_with_tools(tools, temperature) – bind_tools done PER KEY first,
                                              then wrapped with fallbacks
                                              (correct order so each fallback
                                              still knows about the tools)

Caching: results are cached so we don't rebuild N HTTP clients on every node call.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from logger import get_logger


MODEL_NAME = os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")

log = get_logger("gemini")


# ═══════════════════════════════════════════════════════════════════
#  KEY COLLECTION
# ═══════════════════════════════════════════════════════════════════

def _collect_keys() -> list[str]:
    """Return the ordered list of available Gemini API keys.

    Looks at GOOGLE_API_KEY1, GOOGLE_API_KEY2, ... in order until it finds an
    unset value (stops, doesn't skip gaps). Then appends GOOGLE_API_KEY
    (un-numbered) as a final fallback if present and not already listed.
    """
    keys: list[str] = []
    i = 1
    while True:
        v = os.getenv(f"GOOGLE_API_KEY{i}", "").strip()
        if not v:
            break
        keys.append(v)
        i += 1

    single = os.getenv("GOOGLE_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)

    return keys


def available_keys_count() -> int:
    return len(_collect_keys())


# ═══════════════════════════════════════════════════════════════════
#  LLM FACTORY
# ═══════════════════════════════════════════════════════════════════

def _build_one(key: str, temperature: float, max_tokens: Optional[int]) -> ChatGoogleGenerativeAI:
    kwargs = {
        "model":          MODEL_NAME,
        "google_api_key": key,
        "temperature":    temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatGoogleGenerativeAI(**kwargs)


@lru_cache(maxsize=8)
def _make_llm_cached(temperature: float, max_tokens: Optional[int]) -> Runnable:
    keys = _collect_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API key found. Set GOOGLE_API_KEY1 (and optionally "
            "GOOGLE_API_KEY2, KEY3, ...) in .env"
        )

    llms = [_build_one(k, temperature, max_tokens) for k in keys]

    log.info("Built %d Gemini LLM(s)  temperature=%s  max_tokens=%s  fallbacks=%d",
             len(llms), temperature, max_tokens, max(0, len(llms) - 1))

    if len(llms) == 1:
        return llms[0]
    # Primary tries first, others used in order on exception
    return llms[0].with_fallbacks(llms[1:])


def make_llm(temperature: float = 0.3, max_tokens: Optional[int] = None) -> Runnable:
    """Plain chat LLM (no tools) with key fallback."""
    return _make_llm_cached(temperature, max_tokens)


# bind_tools must happen BEFORE with_fallbacks, otherwise the fallback wrapper
# loses tool-binding information. So we keep a separate builder for tool chains.
# We cache by (tools id, temperature) — using id is OK because TOOLS lists are
# module-level constants in each agent.

_tool_chain_cache: dict[tuple[int, float], Runnable] = {}


def make_llm_with_tools(tools: list, temperature: float = 0.3) -> Runnable:
    """Build a Gemini chain with tools bound + key fallback."""
    cache_key = (id(tools), temperature)
    if cache_key in _tool_chain_cache:
        return _tool_chain_cache[cache_key]

    keys = _collect_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API key found. Set GOOGLE_API_KEY1 (and optionally "
            "GOOGLE_API_KEY2, KEY3, ...) in .env"
        )

    raw_llms = [_build_one(k, temperature, max_tokens=None) for k in keys]
    bound    = [llm.bind_tools(tools) for llm in raw_llms]

    log.info("Built tool chain (%d keys, %d tools, temperature=%s)",
             len(bound), len(tools), temperature)

    chain = bound[0] if len(bound) == 1 else bound[0].with_fallbacks(bound[1:])
    _tool_chain_cache[cache_key] = chain
    return chain
