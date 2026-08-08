"""Fail-closed bridge for unrestricted graph runtime/objective requests.

Selected Alice-Bob graph routes are handled by
``sequence_game.repeater_runtime.graph_to_sequence``. This module remains the
guard for callers that ask for an unrestricted generated-graph execution or a
final graph objective claim.
"""

from __future__ import annotations

from typing import Any, Mapping

from .claims import (
    BLOCKED_PENDING_GRAPH_INTERFACE,
    blocked_result,
    validate_blocked_response,
)

GENERATED_GRAPH_RUNTIME_BLOCK = (
    "unrestricted generated graph execution / graph-objective claims remain "
    "blocked; compile and run an explicit selected Alice-Bob route instead"
)


def block_generated_graph_repeater_runtime(
        topology: Mapping[str, Any] | Any,
        *,
        reason: str = GENERATED_GRAPH_RUNTIME_BLOCK) -> dict[str, Any]:
    """Return a blocked response for arbitrary graph runtime requests.

    The topology is accepted only so callers can use one guard function at the
    graph/runtime boundary.  It is not interpreted as a simulator topology here.
    """

    _ = topology
    response = blocked_result(
        reason,
        claim_status=BLOCKED_PENDING_GRAPH_INTERFACE,
        extra={"runtime_supported": False},
    )
    validate_blocked_response(response)
    return response
