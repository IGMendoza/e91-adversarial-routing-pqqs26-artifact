"""Public observation model for Eve.

Builds the (configurably redacted) observation Eve receives after each trial,
from the PublicTranscript only — never from TrialTranscript or simulator
internals. Bucketing is deterministic so observations can key tabular RL
states.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Sequence

from ..protocol.transcript import PublicTranscript

UNKNOWN_TOKEN = "unknown"


class ObservationError(ValueError):
    """Invalid observation configuration or inputs."""


@dataclass(frozen=True)
class ObservationConfig:
    expose_route_id: bool = False
    expose_qber: bool = False
    expose_latency: bool = False
    expose_sifted_count: bool = False
    # ascending interior bucket edges; value x falls in bucket i where
    # edges[i-1] <= x < edges[i] (b0 is x < edges[0], last bucket is open-ended)
    qber_bucket_edges: tuple[float, ...] = ()
    latency_bucket_edges_ps: tuple[int, ...] = ()
    unknown_token: str = UNKNOWN_TOKEN

    def __post_init__(self) -> None:
        for name, edges in (("qber_bucket_edges", self.qber_bucket_edges),
                            ("latency_bucket_edges_ps", self.latency_bucket_edges_ps)):
            if list(edges) != sorted(edges) or len(set(edges)) != len(edges):
                raise ObservationError(f"{name} must be strictly ascending")
        if self.expose_qber and not self.qber_bucket_edges:
            raise ObservationError("expose_qber requires qber_bucket_edges")
        if self.expose_latency and not self.latency_bucket_edges_ps:
            raise ObservationError("expose_latency requires latency_bucket_edges_ps")


def bucket_value(value: Optional[float], edges: Sequence[float],
                 unknown_token: str = UNKNOWN_TOKEN) -> str:
    """Deterministic bucket label: b0..b{len(edges)} or the unknown token."""
    if value is None:
        return unknown_token
    index = 0
    for edge in edges:
        if value >= edge:
            index += 1
        else:
            break
    return f"b{index}"


@dataclass(frozen=True)
class PublicObservation:
    previous_action_id: str
    outcome: str  # "accept" | "abort" | unknown token
    route_id: str
    qber_bucket: str
    latency_bucket: str
    sifted_count: str

    def as_state_key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, f.name)) for f in fields(self))

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def build_observation(public: PublicTranscript, previous_action_id: str,
                      config: ObservationConfig) -> PublicObservation:
    unk = config.unknown_token
    if public.accepted is None:
        outcome = unk
    else:
        outcome = "accept" if public.accepted else "abort"
    route_id = public.route_id if (config.expose_route_id and public.route_id) else unk
    qber_bucket = (bucket_value(public.qber_estimate, config.qber_bucket_edges, unk)
                   if config.expose_qber else unk)
    latency_bucket = (bucket_value(public.latency_ps, config.latency_bucket_edges_ps, unk)
                      if config.expose_latency else unk)
    sifted = (str(public.sifted_count)
              if config.expose_sifted_count and public.sifted_count is not None else unk)
    return PublicObservation(
        previous_action_id=previous_action_id,
        outcome=outcome,
        route_id=route_id,
        qber_bucket=qber_bucket,
        latency_bucket=latency_bucket,
        sifted_count=sifted,
    )
