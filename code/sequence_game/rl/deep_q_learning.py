"""Deep Q-Network scaffold with identical interface to QTable.

Uses one sklearn MLPRegressor per action, trained on an experience replay
buffer.  The network refits every `refit_every` steps once the buffer has
enough samples; before the first refit all Q-values are 0.0.

State encoding: per-position string vocabulary → one-hot; positions are
grown dynamically so any tuple of strings works.  This generalises to
larger state spaces (full graph state, fidelity history) without changes
to the interface.

to_dict() enumerates all seen states and returns their current Q-values
so the surrogate DT pipeline can treat DeepQNetwork output identically to
a QTable.  The format matches QTable.to_dict() exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Above this many exported Q entries (states x actions), to_dict() stops
# enumerating the full vocabulary cross-product and falls back to states the
# agent actually visited, truncating to the most recent ones if needed.
# Route-dense corpus graphs (~30k Alice route actions) otherwise materialize
# multi-GB exports and OOM the corpus worker.
MAX_DENSE_Q_EXPORT_ENTRIES = 2_000_000


class _StateEncoder:
    """Map tuples of strings to fixed-max-size one-hot float vectors.

    Each position in the state tuple gets its own vocabulary; all positions
    are concatenated.  When the vocabulary grows the feature vector expands;
    stored state vectors are padded with zeros automatically.
    """

    def __init__(self) -> None:
        self._vocabs: list[dict[str, int]] = []

    def encode(self, state: tuple) -> np.ndarray:
        parts: list[float] = []
        for i, token in enumerate(state):
            token = str(token)
            if i >= len(self._vocabs):
                self._vocabs.append({})
            vocab = self._vocabs[i]
            if token not in vocab:
                vocab[token] = len(vocab)
            oh = [0.0] * len(vocab)
            oh[vocab[token]] = 1.0
            parts.extend(oh)
        return np.array(parts, dtype=float)

    @property
    def feature_size(self) -> int:
        return sum(len(v) for v in self._vocabs)

    def vocab_product_size(self) -> int:
        """Size of the full state cross-product without materializing it."""
        if not self._vocabs:
            return 0
        size = 1
        for vocab in self._vocabs:
            size *= max(1, len(vocab))
        return size

    def all_state_tuples(self) -> list[tuple]:
        """Enumerate all observed state tuples (for to_dict)."""
        import itertools
        if not self._vocabs:
            return []
        keys_per_pos = [list(v.keys()) for v in self._vocabs]
        return list(itertools.product(*keys_per_pos))


class DeepQNetwork:
    """Function-approximation Q-learner; identical interface to QTable.

    Scaffold for non-interpretable policies.  The XAI pipeline detects
    ``interpretable=False`` in POLICY_REGISTRY and fits a surrogate DT
    on the Q-values returned by to_dict().
    """

    def __init__(
        self,
        num_actions: int,
        hidden: tuple[int, ...] = (64, 32),
        refit_every: int = 50,
        buffer_size: int = 500,
    ) -> None:
        self.num_actions = num_actions
        self._hidden = hidden
        self._refit_every = refit_every
        self._max_buffer = buffer_size
        self._encoder = _StateEncoder()
        # One regressor per action; None until enough data for first refit
        self._models: list[Optional[object]] = [None] * num_actions
        # Replay buffer: (state_tuple, action_idx, target_q_float)
        self._buffer: list[tuple[tuple, int, float]] = []
        # Insertion-ordered set of states actually passed to update()
        self._seen_states: dict[tuple, None] = {}
        self._step = 0

    # ── Public interface (mirrors QTable) ────────────────────────────────

    def values(self, state: tuple) -> list[float]:
        """Return Q-values for all actions at this state."""
        state = tuple(str(s) for s in state)
        enc = self._encoder.encode(state).reshape(1, -1)
        return [self._predict_action(a, enc) for a in range(self.num_actions)]

    def get(self, state: tuple, action: int) -> float:
        return self.values(state)[action]

    def best_action(
        self,
        state: tuple,
        allowed_actions: Optional[Sequence[int]] = None,
    ) -> int:
        vals = self.values(state)
        allowed = list(allowed_actions) if allowed_actions is not None else list(range(self.num_actions))
        return min(allowed, key=lambda i: (-vals[i], i))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: Optional[tuple],
        config: object,
    ) -> float:
        """One-step experience-replay update; returns the target Q-value."""
        state  = tuple(str(s) for s in state)
        # Encode state now so its tokens enter the vocabulary even when next_state is None
        self._encoder.encode(state)
        self._seen_states[state] = None
        if next_state is not None:
            self._seen_states[tuple(str(s) for s in next_state)] = None
        next_q = max(self.values(next_state)) if next_state is not None else 0.0
        target    = reward + getattr(config, "gamma", 0.9) * next_q
        self._buffer.append((state, action, target))
        if len(self._buffer) > self._max_buffer:
            self._buffer.pop(0)
        self._step += 1
        if self._step % self._refit_every == 0:
            self._refit()
        return target

    def seen_state_tuples(self) -> list[tuple]:
        """States actually passed through update(), in first-seen order."""
        return list(self._seen_states)

    def to_dict(self) -> dict:
        """Return seen states with current Q-values — matches QTable.to_dict() schema.

        Small state spaces keep the historical vocabulary cross-product
        enumeration; past MAX_DENSE_Q_EXPORT_ENTRIES the export is bounded to
        the most recently visited states and flagged with q_export metadata.
        """
        product_size = self._encoder.vocab_product_size()
        export_meta: Optional[dict] = None
        if product_size * self.num_actions <= MAX_DENSE_Q_EXPORT_ENTRIES:
            states = self._encoder.all_state_tuples()
        else:
            states = list(self._seen_states)
            keep = max(1, MAX_DENSE_Q_EXPORT_ENTRIES // max(1, self.num_actions))
            truncated = len(states) > keep
            if truncated:
                states = states[-keep:]
            export_meta = {
                "mode": "seen_states",
                "truncated": truncated,
                "seen_state_count": len(self._seen_states),
                "vocab_product_size": product_size,
            }
        entries = []
        for state_tuple in states:
            entries.append({
                "state": list(state_tuple),
                "q": self.values(state_tuple),
            })
        payload = {"num_actions": self.num_actions, "entries": entries}
        if export_meta is not None:
            payload["q_export"] = export_meta
        return payload

    # ── Internal ─────────────────────────────────────────────────────────

    def _predict_action(self, action: int, enc: np.ndarray) -> float:
        model = self._models[action]
        if model is None:
            return 0.0
        try:
            feat = enc
            # Pad if vocabulary grew since last refit
            expected = model.n_features_in_
            if feat.shape[1] < expected:
                pad = np.zeros((1, expected - feat.shape[1]))
                feat = np.concatenate([feat, pad], axis=1)
            elif feat.shape[1] > expected:
                feat = feat[:, :expected]
            return float(model.predict(feat)[0])
        except Exception:
            return 0.0

    def _make_model(self):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=self._hidden,
            solver="lbfgs",   # quasi-Newton: convergent on small datasets, no SGD noise
            max_iter=400,
            random_state=42,
            warm_start=True,  # retain weights across refits — avoids re-learning from scratch
        )

    def _refit(self) -> None:
        # Single pass over the buffer: scanning range(num_actions) is O(A*B)
        # and A reaches ~30k on route-dense corpus graphs.
        samples_by_action: dict[int, list[tuple[tuple, float]]] = {}
        for s, a, t in self._buffer:
            samples_by_action.setdefault(a, []).append((s, t))
        for action in sorted(samples_by_action):
            samples = samples_by_action[action]
            if len(samples) < 4:
                continue
            states, targets = zip(*samples)
            # Vocabulary may grow as we encode; pad rows to uniform max length
            encoded = [self._encoder.encode(s) for s in states]
            max_len = max(len(e) for e in encoded)
            X = np.zeros((len(encoded), max_len))
            for i, e in enumerate(encoded):
                X[i, :len(e)] = e
            y = np.array(targets, dtype=float)
            # Reuse the existing model so warm_start continues from prior weights
            if self._models[action] is None:
                self._models[action] = self._make_model()
            try:
                self._models[action].fit(X, y)
            except Exception:
                self._models[action] = None
