#!/usr/bin/env python
"""Compatibility entry point for :mod:`histograms.cli`.

New code should import the focused modules under ``histograms``.  Attribute
access remains proxied to the historical engine so existing diagnostics and
tests that imported helper functions from this script continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from histograms import engine as _engine


class _EngineProxy(ModuleType):
    def __getattr__(self, name: str) -> Any:
        return getattr(_engine, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(_engine)))

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in {"__class__", "__dict__"} and hasattr(_engine, name):
            setattr(_engine, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _EngineProxy


if __name__ == "__main__":
    raise SystemExit(_engine.main())
