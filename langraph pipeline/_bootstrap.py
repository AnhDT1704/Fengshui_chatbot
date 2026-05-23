"""
_bootstrap.py – Make the parent project directory importable.

The folder containing the chatbot files (`langraph pipeline/`) has a space
in its name, so we can't use it as a Python package. Every chatbot module
imports this first to add the parent directory (which contains
db_service / opensearch_service / embedding_service / models / config)
to sys.path.
"""

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
