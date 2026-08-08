"""First-party pre-swap memory intervention hook for entanglement swapping.

This module reintroduces the pre-swap memory hook that used to live in the
SeQUeNCe submodule patches (``sequence-submodule-2-commits.patch``). Upstream
SeQUeNCe v1.0.0 turned entanglement swapping into a registry-based package, so
the hook can now be added entirely on the first-party side of the boundary: we
register an ``EntanglementSwappingA`` subclass that fires an optional callback
immediately before the swapping circuit runs, then defers to the stock circuit
swap.

Design notes:

* The hook is looked up per-node via an attribute on the owning router
  (:data:`PRE_SWAP_HOOK_ATTR`). Router node objects are created fresh per
  ``RouterNetTopo`` build, so there is no cross-trial global leak: an installed
  hook only affects the trial whose nodes carry it.
* The subclass behaves identically to the stock circuit swap when no hook is
  installed, so leaving it as the active swapping formalism is behavior
  preserving. Callers should still scope :func:`use_pre_swap_hook_swapping`
  around the timeline run for cleanliness.
* The hook fires only on a successful swap, at the exact point the stock
  protocol consumes the memories -- matching the retired submodule patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from sequence.components.circuit import Circuit
from sequence.constants import DENSITY_MATRIX_FORMALISM, KET_VECTOR_FORMALISM
from sequence.entanglement_management.swapping import EntanglementSwappingA
from sequence.entanglement_management.swapping.swapping_base import (
    EntanglementSwappingMessage,
    SwappingMsgType,
)
from sequence.entanglement_management.swapping.swapping_circuit import (
    EntanglementSwappingA_Circuit,
)
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.utils import log

PRE_SWAP_HOOK_SWAPPING = "circuit_pre_swap_hook"
PRE_SWAP_HOOK_ATTR = "_pre_swap_memory_hook"


@dataclass(frozen=True)
class SwapContext:
    """Context passed to an optional pre-swap memory intervention hook.

    Ported verbatim from the retired submodule patch so downstream hook code
    (measurement / dephasing probe) keeps the same interface.
    """

    repeater_node_name: str
    left_memory_name: str
    left_qstate_key: int
    right_memory_name: str
    right_qstate_key: int
    quantum_manager: Any
    timeline_time: int
    swap_protocol: Any
    route_metadata: Any = None


@EntanglementSwappingA.register(PRE_SWAP_HOOK_SWAPPING)
class EntanglementSwappingA_PreSwapHook(EntanglementSwappingA_Circuit):
    """Circuit swap that fires an optional per-node pre-swap memory hook.

    The hook, if any, is read from :data:`PRE_SWAP_HOOK_ATTR` on the owning
    node. It is invoked immediately before the swapping circuit consumes the
    two memories, only when the swap succeeds -- the same placement as the
    former submodule patch on ``EntanglementSwappingA``.

    The body mirrors ``EntanglementSwappingA_Circuit.start`` (SeQUeNCe v1.0.0)
    with the hook inserted; keep it in sync if the upstream circuit swap
    changes. The stock metrics recording is intentionally omitted here because
    it is optional diagnostics; the swap physics (fidelity, measurement,
    messaging) are identical.
    """

    def start(self) -> None:
        log.logger.info(
            f"{self.owner.name} middle protocol start with ends "
            f"{self.left_node}, {self.right_node}")

        assert self.left_memo.fidelity > 0 and self.right_memo.fidelity > 0
        assert self.left_memo.entangled_memory["node_id"] == self.left_node
        assert self.right_memo.entangled_memory["node_id"] == self.right_node

        if self.owner.get_generator().random() < self.success_probability():
            # swapping succeeded
            fidelity = self.updated_fidelity(self.left_memo.fidelity, self.right_memo.fidelity)
            self.is_success = True

            expire_time = min(self.left_memo.get_expire_time(),
                              self.right_memo.get_expire_time())

            hook = getattr(self.owner, PRE_SWAP_HOOK_ATTR, None)
            if hook is not None:
                hook(SwapContext(
                    repeater_node_name=self.owner.name,
                    left_memory_name=self.left_memo.name,
                    left_qstate_key=self.left_memo.qstate_key,
                    right_memory_name=self.right_memo.name,
                    right_qstate_key=self.right_memo.qstate_key,
                    quantum_manager=self.owner.timeline.quantum_manager,
                    timeline_time=self.owner.timeline.now(),
                    swap_protocol=self,
                    route_metadata=None,
                ))

            meas_samp = self.owner.get_generator().random()
            meas_res = self.owner.timeline.quantum_manager.run_circuit(
                self.circuit,
                [self.left_memo.qstate_key, self.right_memo.qstate_key],
                meas_samp)
            meas_res = [meas_res[self.left_memo.qstate_key],
                        meas_res[self.right_memo.qstate_key]]

            log.logger.info(
                f"{self.name} swapping succeeded, meas_res={meas_res[0]},{meas_res[1]}")

            msg_l = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES, self.left_protocol_name, fidelity=fidelity,
                remote_node=self.right_memo.entangled_memory["node_id"],
                remote_memo=self.right_memo.entangled_memory["memo_id"],
                expire_time=expire_time, meas_res=[])
            msg_r = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES, self.right_protocol_name, fidelity=fidelity,
                remote_node=self.left_memo.entangled_memory["node_id"],
                remote_memo=self.left_memo.entangled_memory["memo_id"],
                expire_time=expire_time, meas_res=meas_res)
        else:
            # swapping failed
            log.logger.info(f"{self.name} swapping failed")
            msg_l = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES, self.left_protocol_name, fidelity=0)
            msg_r = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES, self.right_protocol_name, fidelity=0)

        self.owner.send_message(self.left_node, msg_l)
        self.owner.send_message(self.right_node, msg_r)

        self.update_resource_manager(self.left_memo, MemoryInfo.RAW)
        self.update_resource_manager(self.right_memo, MemoryInfo.RAW)

    @lru_cache(maxsize=128)
    def updated_fidelity(self, f1: float, f2: float) -> float:
        return f1 * f2 * self.degradation


def install_pre_swap_hook(node: Any, hook: Callable[[SwapContext], None] | None) -> None:
    """Attach (or clear) a pre-swap memory hook on a router node."""

    setattr(node, PRE_SWAP_HOOK_ATTR, hook)


class use_pre_swap_hook_swapping:
    """Context manager that scopes the pre-swap-hook swapping formalism.

    Sets :data:`PRE_SWAP_HOOK_SWAPPING` as the active ``EntanglementSwappingA``
    formalism for the duration of the ``with`` block and restores the previous
    formalism afterwards. The paired ``EntanglementSwappingB`` is unchanged
    (the hook only touches the middle-node A protocol), so only the A formalism
    is swapped.
    """

    def __enter__(self) -> "use_pre_swap_hook_swapping":
        self._previous = EntanglementSwappingA.get_formalism()
        EntanglementSwappingA.set_formalism(PRE_SWAP_HOOK_SWAPPING)
        return self

    def __exit__(self, *_exc: object) -> None:
        EntanglementSwappingA.set_formalism(self._previous)


# Sanity: our custom formalism registered under both the ket-vector and density
# matrix default circuit slots would collide, so we only register the custom
# name. Confirm the stock circuit formalisms still exist for restore.
assert KET_VECTOR_FORMALISM in EntanglementSwappingA.list_protocols()
assert DENSITY_MATRIX_FORMALISM in EntanglementSwappingA.list_protocols()
