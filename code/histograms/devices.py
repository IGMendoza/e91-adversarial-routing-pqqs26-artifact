"""Compatibility surface for decomposed physical-device components.

Prefer importing ``device_models``, ``device_profiles``, ``device_effects``,
or ``front_matter`` directly in new code.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import device_effects, device_models, device_profiles, front_matter

_COMPONENTS = (device_models, device_profiles, device_effects, front_matter)
_OWNER_BY_NAME: dict[str, ModuleType] = {}
for _component in _COMPONENTS:
    for _name in vars(_component):
        if _name.startswith("__"):
            continue
        _OWNER_BY_NAME.setdefault(_name, _component)

__all__ = sorted(name for name in _OWNER_BY_NAME if not name.startswith("_"))


class _DevicesModule(ModuleType):
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


sys.modules[__name__].__class__ = _DevicesModule
