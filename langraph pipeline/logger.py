"""
logger.py – Central logging for the chatbot.

Outputs to BOTH the console (colored) and a rotating file at
`<project_root>/logs/chatbot.log`.

Usage:
    from logger import get_logger
    log = get_logger("kb")
    log.info("hello %s", name)

Tail the log (PowerShell):
    Get-Content -Wait logs\\chatbot.log

Filter (PowerShell):
    Get-Content -Wait logs\\chatbot.log | Select-String -Pattern "TOOL"
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Resolve the project root (parent of "langraph pipeline/")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR  = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "chatbot.log"
LOG_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("CHATBOT_LOG_LEVEL", "INFO").upper()


# ═══════════════════════════════════════════════════════════════════
#  COLOR FORMATTER (console only)
# ═══════════════════════════════════════════════════════════════════

class _ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    NAME_COLORS = {
        "supervisor":   "\033[95m",   # bright magenta
        "small_talk":   "\033[94m",   # bright blue
        "kb":           "\033[94m",   # bright blue
        "skills":       "\033[96m",   # bright cyan
        "vision":       "\033[93m",   # bright yellow
        "order":        "\033[92m",   # bright green
        "tool":         "\033[37m",   # white
        "memory":       "\033[90m",   # bright black/gray
        "graph":        "\033[97m",   # bright white
        "api":          "\033[90m",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not _color_enabled():
            return base
        lvl = self.COLORS.get(record.levelname, "")

        # Colorize the logger name segment if we can find it
        for short, ncolor in self.NAME_COLORS.items():
            tag = f"[{short}]"
            if tag in base:
                base = base.replace(tag, f"{ncolor}{tag}{self.RESET}")
                break

        return f"{lvl}{base}{self.RESET}" if lvl else base


def _color_enabled() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════

_LOGGER_NAMESPACE = "chatbot"
_initialized = False


def _setup_root() -> logging.Logger:
    global _initialized
    root = logging.getLogger(_LOGGER_NAMESPACE)
    if _initialized:
        return root

    root.setLevel(LOG_LEVEL)
    root.propagate = False

    fmt     = "%(asctime)s %(levelname)-7s [%(short_name)s] %(message)s"
    datefmt = "%H:%M:%S"

    # Inject short_name attribute (strip "chatbot." prefix)
    class _ShortNameFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            full = record.name
            record.short_name = full.split(".", 1)[1] if "." in full else full
            return True

    name_filter = _ShortNameFilter()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ColorFormatter(fmt, datefmt))
    ch.addFilter(name_filter)
    root.addHandler(ch)

    # Rotating file handler (5 MB × 3 backups)
    try:
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        fh.addFilter(name_filter)
        root.addHandler(fh)
    except Exception as e:
        sys.stderr.write(f"[logger] could not open file handler: {e}\n")

    _initialized = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the chatbot namespace."""
    _setup_root()
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")


# ═══════════════════════════════════════════════════════════════════
#  TOOL CALLBACK  (LangChain callback handler for tool invocation)
# ═══════════════════════════════════════════════════════════════════

try:
    from langchain_core.callbacks import BaseCallbackHandler

    class ToolLoggerCallback(BaseCallbackHandler):
        """LangChain callback that logs every tool call.

        Attach via `graph.invoke(state, config={"callbacks": [ToolLoggerCallback("kb")]})`.
        """

        def __init__(self, agent_name: str = "tool"):
            self.log = get_logger(f"tool.{agent_name}")

        def on_tool_start(
            self,
            serialized: dict,
            input_str: str,
            *,
            run_id=None,
            parent_run_id=None,
            tags: Optional[list[str]] = None,
            metadata: Optional[dict] = None,
            inputs: Optional[dict] = None,
            **kwargs: Any,
        ) -> None:
            name = (serialized or {}).get("name", "?")
            args_repr = (input_str or str(inputs or {}))[:200]
            self.log.info("CALL  %s(%s)", name, args_repr)

        def on_tool_end(
            self,
            output: Any,
            *,
            run_id=None,
            parent_run_id=None,
            **kwargs: Any,
        ) -> None:
            snippet = str(output)[:200].replace("\n", " ")
            self.log.debug("OK    -> %s", snippet)

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id=None,
            parent_run_id=None,
            **kwargs: Any,
        ) -> None:
            self.log.error("ERROR -> %s: %s", type(error).__name__, error)

except ImportError:   # LangChain not available — degrade gracefully
    class ToolLoggerCallback:                              # type: ignore[no-redef]
        def __init__(self, *_a, **_k): pass
