"""Detect a "probe, back off, strike" bait-and-switch pattern in Exp3 self-play.

On a topology with multiple node-disjoint Alice-Bob paths, Eve can in
principle learn to withhold attack on whichever route Alice's policy is
converging onto -- letting her weight ratchet up further because the route
looks unattacked and safe -- then increase attack pressure once Alice's
commitment is maximal. This module operationalizes a mechanical check for
that pattern (a lagged-correlation reversal) against real, already-logged
per-turn strategy trajectories from an ``exp3_vs_exp3`` online run, and a
model-agnostic check of whether pooled graph-shape features predict which
graphs exhibit it.

Scientific scope: toy / descriptive statistics on real logged trajectories
and real feature correlations. No new equations, security claims, or
physical models -- this module answers a detectability question, not a
security-guarantee question. See ``GRAPH_SHAPE_XAI_GROUNDING.md``'s
"Border collie" section for the full real-corpus results this produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


def _route_edge_labels(route: dict[str, Any]) -> set[str]:
    """Same normalization as ``exp3_sequence/dt_rows.py``'s private helper
    of the same name (duplicated per that module's own precedent of
    keeping this tiny helper local rather than centralizing it)."""
    path = [str(node) for node in route.get("path", [])]
    labels = {
        f"{min(u, v)}-{max(u, v)}"
        for u, v in zip(path, path[1:])
    }
    labels.update(str(edge_id) for edge_id in route.get("edge_ids", ()))
    return labels


def action_covers_route(action_id: str, route: dict[str, Any]) -> bool:
    """Does attacking ``action_id`` plausibly deny ``route``?

    Same coverage definition as ``target_route_coverage`` in ``dt_rows.py``:
    an edge-kind action whose target edge lies on the route, or a node-kind
    action whose target node is an internal node of the route. ``no_attack``
    never covers anything.
    """
    if action_id == "no_attack":
        return False
    kind, _, target = action_id.partition(":")
    if kind == "edge_intercept_resend":
        return target in _route_edge_labels(route)
    if kind == "memory_degradation":
        return target in set(route.get("internal_nodes") or [])
    return False


@dataclass
class BorderCollieResult:
    graph_id: str
    num_routes: int
    num_actions: int
    favorite_route_id: str
    favorite_final_weight: float
    eve_weight_early_mean: float
    eve_weight_late_mean: float
    alice_weight_early_mean: float
    alice_weight_late_mean: float
    corr_early: float
    corr_late: float
    score: float  # corr_late - corr_early; large positive = bait-and-switch signature


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(stats.pearsonr(a, b)[0])


def compute_border_collie_result(
    graph_id: str,
    routes: list[dict[str, Any]],
    action_ids: list[str],
    steps: list[dict[str, Any]],
) -> BorderCollieResult | None:
    """Compute the lagged-correlation border-collie score for one graph.

    ``steps`` is the per-turn snapshot list already logged by the online
    runner (each with ``turn``, ``alice_strategy``, ``eve_strategy`` --
    see ``sequence_game/experiments/exp3_sequence/online.py``). Returns
    ``None`` when there isn't enough data to compute a meaningful score
    (too few routes, too few snapshots, or no action covers Alice's
    favorite route), matching this module's real-corpus usage where a
    minority of candidate graphs are skipped for exactly these reasons.
    """
    if len(steps) < 20 or len(routes) < 3:
        return None

    turns = np.array([s["turn"] for s in steps], dtype=float)
    alice_mat = np.array([s["alice_strategy"] for s in steps])
    eve_mat = np.array([s["eve_strategy"] for s in steps])
    if alice_mat.shape[1] != len(routes) or eve_mat.shape[1] != len(action_ids):
        return None

    favorite_idx = int(np.argmax(alice_mat[-1]))
    favorite_route = routes[favorite_idx]
    favorite_final_weight = float(alice_mat[-1, favorite_idx])

    covering = [i for i, aid in enumerate(action_ids) if action_covers_route(aid, favorite_route)]
    if not covering:
        return None

    alice_series = alice_mat[:, favorite_idx]
    eve_series = eve_mat[:, covering].sum(axis=1)

    half = len(turns) // 2
    if half < 5:
        return None

    corr_early = _safe_pearson(alice_series[:half], eve_series[:half])
    corr_late = _safe_pearson(alice_series[half:], eve_series[half:])

    return BorderCollieResult(
        graph_id=graph_id,
        num_routes=len(routes),
        num_actions=len(action_ids),
        favorite_route_id=str(favorite_route.get("route_id")),
        favorite_final_weight=favorite_final_weight,
        eve_weight_early_mean=float(eve_series[:half].mean()),
        eve_weight_late_mean=float(eve_series[half:].mean()),
        alice_weight_early_mean=float(alice_series[:half].mean()),
        alice_weight_late_mean=float(alice_series[half:].mean()),
        corr_early=corr_early,
        corr_late=corr_late,
        score=corr_late - corr_early,
    )


# ---------------------------------------------------------------------------
# Stricter operationalization: brief, repeated Eve probes that Alice's own
# learning treats as noise, rather than a sustained regime change. Added
# after inspecting the lagged-correlation score's top examples directly:
# none showed Eve's attack weight going from genuinely LOW early to HIGH
# late in absolute terms -- the correlation-sign metric above is real but
# doesn't by itself isolate the "brief strike, absorbed as a fluke" story.
# This second detector targets that more specific mechanism directly.
# ---------------------------------------------------------------------------

DEFAULT_SPIKE_K_MAD = 3.0
DEFAULT_MAX_SPIKE_SNAPSHOTS = 2
DEFAULT_MIN_BRIEF_EVENTS = 2


@dataclass
class ProbeAndShrugResult:
    graph_id: str
    favorite_route_id: str
    num_spike_events: int
    num_brief_spike_events: int
    longest_event_snapshots: int
    alice_net_trend: float
    alice_final_weight: float
    is_probe_and_shrug_candidate: bool


def find_spike_events(series: np.ndarray, k_mad: float = DEFAULT_SPIKE_K_MAD) -> list[tuple[int, int]]:
    """Maximal contiguous runs of ``series`` above a robust per-series
    baseline (median + ``k_mad`` * MAD).

    A robust (median/MAD, not mean/std) baseline is used because Exp3's
    own gamma-exploration floor means a covering action's weight is never
    exactly zero even absent any deliberate strategy -- a robust baseline
    is less sensitive to that floor and to the occasional large excursion
    than a mean-based one would be. Falls back to sample std when MAD is
    degenerate (a perfectly flat series), so a constant series never
    divides by zero.
    """
    baseline = float(np.median(series))
    mad = float(np.median(np.abs(series - baseline)))
    if mad < 1e-9:
        mad = float(np.std(series)) or 1e-9
    threshold = baseline + k_mad * mad
    above = series > threshold

    events: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(series) - 1))
    return events


def compute_probe_and_shrug_result(
    graph_id: str,
    routes: list[dict[str, Any]],
    action_ids: list[str],
    steps: list[dict[str, Any]],
    *,
    k_mad: float = DEFAULT_SPIKE_K_MAD,
    max_spike_snapshots: int = DEFAULT_MAX_SPIKE_SNAPSHOTS,
    min_brief_events: int = DEFAULT_MIN_BRIEF_EVENTS,
) -> ProbeAndShrugResult | None:
    """Detect brief, repeated Eve attack-weight spikes on Alice's favorite
    route, alongside Alice's own weight still net-rising across the run
    (evidence she treated the incidents as noise rather than a threat).

    Returns ``None`` under the same insufficient-data conditions as
    :func:`compute_border_collie_result` (too few routes/snapshots, or no
    action covers Alice's favorite route) -- these two detectors are
    typically run over the same candidate graphs.
    """
    if len(steps) < 20 or len(routes) < 3:
        return None

    alice_mat = np.array([s["alice_strategy"] for s in steps])
    eve_mat = np.array([s["eve_strategy"] for s in steps])
    if alice_mat.shape[1] != len(routes) or eve_mat.shape[1] != len(action_ids):
        return None

    favorite_idx = int(np.argmax(alice_mat[-1]))
    favorite_route = routes[favorite_idx]

    covering = [i for i, aid in enumerate(action_ids) if action_covers_route(aid, favorite_route)]
    if not covering:
        return None

    alice_series = alice_mat[:, favorite_idx]
    eve_series = eve_mat[:, covering].sum(axis=1)

    events = find_spike_events(eve_series, k_mad)
    brief_events = [(s, e) for s, e in events if (e - s + 1) <= max_spike_snapshots]
    alice_net_trend = float(alice_series[-1] - alice_series[0])

    return ProbeAndShrugResult(
        graph_id=graph_id,
        favorite_route_id=str(favorite_route.get("route_id")),
        num_spike_events=len(events),
        num_brief_spike_events=len(brief_events),
        longest_event_snapshots=max((e - s + 1 for s, e in events), default=0),
        alice_net_trend=alice_net_trend,
        alice_final_weight=float(alice_series[-1]),
        is_probe_and_shrug_candidate=(len(brief_events) >= min_brief_events and alice_net_trend > 0),
    )
