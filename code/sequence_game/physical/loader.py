"""JSON loaders for physical-model configs.

Loading is structural only. ``require_resolved=True`` (the posture for any
non-toy run) rejects configs that still carry TODO/None values, so that
unresolved literature configs can never silently flow into an experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .registry import PhysicalModel, PhysicalRegistry


def load_physical_model(path: Union[str, Path], *, require_resolved: bool = False) -> PhysicalModel:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = PhysicalModel.from_dict(data)
    if require_resolved:
        model.require_resolved()
    return model


def load_models_from_dir(directory: Union[str, Path], *,
                         require_resolved: bool = False) -> PhysicalRegistry:
    """Load every ``*.json`` model file in a directory into a registry."""
    registry = PhysicalRegistry()
    for path in sorted(Path(directory).glob("*.json")):
        registry.add_model(load_physical_model(path, require_resolved=require_resolved))
    return registry
