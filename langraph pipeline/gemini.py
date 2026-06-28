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
#  OPENROUTER (primary provider, falls back to Google keys)
# ═══════════════════════════════════════════════════════════════════

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _openrouter_key() -> str:
    """OpenRouter API key dành RIÊNG cho chatbot LLM (khác key embedding)."""
    return os.getenv("OPENROUTER_LLM_API_KEY", "").strip()


def _openrouter_model() -> str:
    return os.getenv("OPENROUTER_LLM_MODEL", "google/gemini-2.5-flash").strip()


def _build_openrouter(temperature: float, max_tokens: Optional[int]):
    """Build a ChatOpenAI client pointed at OpenRouter (OpenAI-compatible API)."""
    from langchain_openai import ChatOpenAI  # imported lazily so the dep is only

    kwargs = {                               # required when OpenRouter is enabled
        "model":        _openrouter_model(),
        "api_key":      _openrouter_key(),
        "base_url":     OPENROUTER_BASE_URL,
        "temperature":  temperature,
        "default_headers": {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
            "X-Title":      os.getenv("OPENROUTER_SITE_NAME", "Van An Group Chatbot"),
        },
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


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
    """Số provider khả dụng = (1 nếu có OpenRouter) + số key Google."""
    return (1 if _openrouter_key() else 0) + len(_collect_keys())


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
    # Primary = OpenRouter (nếu có key), sau đó fallback lần lượt sang các key
    # Google free khi OpenRouter lỗi/hết credit.
    llms: list = []
    if _openrouter_key():
        llms.append(_build_openrouter(temperature, max_tokens))

    google_keys = _collect_keys()
    llms += [_build_one(k, temperature, max_tokens) for k in google_keys]

    if not llms:
        raise RuntimeError(
            "No LLM key found. Set OPENROUTER_LLM_API_KEY hoặc GOOGLE_API_KEY1 "
            "(và tuỳ chọn GOOGLE_API_KEY2, KEY3, ...) trong .env"
        )

    log.info("Built %d LLM(s)  openrouter_primary=%s  google_fallbacks=%d  temperature=%s  max_tokens=%s",
             len(llms), bool(_openrouter_key()), len(google_keys), temperature, max_tokens)

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

    bound: list = []
    if _openrouter_key():
        bound.append(_build_openrouter(temperature, max_tokens=None).bind_tools(tools))

    google_keys = _collect_keys()
    bound += [_build_one(k, temperature, max_tokens=None).bind_tools(tools) for k in google_keys]

    if not bound:
        raise RuntimeError(
            "No LLM key found. Set OPENROUTER_LLM_API_KEY hoặc GOOGLE_API_KEY1 "
            "(và tuỳ chọn GOOGLE_API_KEY2, KEY3, ...) trong .env"
        )

    log.info("Built tool chain (openrouter_primary=%s, %d google keys, %d tools, temperature=%s)",
             bool(_openrouter_key()), len(google_keys), len(tools), temperature)

    chain = bound[0] if len(bound) == 1 else bound[0].with_fallbacks(bound[1:])
    _tool_chain_cache[cache_key] = chain
    return chain
