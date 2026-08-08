"""Compatibility aggregation for the decomposed histogram implementation.

New code should import the focused modules directly.  This module preserves
the historical single-module attribute surface, including private diagnostic
helpers used by the repository's regression tests, without owning numerical
state or duplicating caches.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import (
    compiled,
    device_effects,
    device_models,
    device_profiles,
    devices,
    fock,
    front_matter,
    generation,
    models,
    output,
    quantum,
    routing,
    timing,
    topology,
)
from . import cli

_COMPONENTS = (
    models,
    device_models,
    device_profiles,
    device_effects,
    front_matter,
    topology,
    quantum,
    timing,
    fock,
    routing,
    output,
    generation,
    compiled,
    cli,
)

_OWNER_BY_NAME: dict[str, ModuleType] = {}
for _component in _COMPONENTS:
    for _name in vars(_component):
        if _name.startswith("__"):
            continue
        # Components are ordered from foundational to orchestration layers.
        # ``setdefault`` keeps an imported alias mapped to the module where it
        # first appeared, which is its real state/function owner.
        _OWNER_BY_NAME.setdefault(_name, _component)

__all__ = sorted(_OWNER_BY_NAME)


class _EngineModule(ModuleType):
    def __getattr__(self, name: str) -> Any:
        try:
            owner = _OWNER_BY_NAME[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return getattr(owner, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(_OWNER_BY_NAME))

    def __setattr__(self, name: str, value: Any) -> None:
        owner = _OWNER_BY_NAME.get(name)
        if owner is not None:
            setattr(owner, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _EngineModule


if __name__ == "__main__":
    raise SystemExit(cli.main())
