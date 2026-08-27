# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Worker-side hop learner (next-hop coordination plane).

A w2w trainer teaches each stage both its **next hop** -- the next-stage worker that should receive
this microbatch's activation (taught on the forward) -- and its **previous hop** -- the prior-stage
worker its gradient flows back to (taught on the explicit backward, and on the fused tail's forward).
Both ride in-band via the per-microbatch
:class:`~agora_server.core.batch_metadata.BatchMetadata` (``next_hop_uid`` / ``next_hop_peer_id`` and
``prev_hop_uid`` / ``prev_hop_peer_id``). :class:`NextHopLearner` is the worker side of that channel:
on every forward AND backward RPC it parses the metadata, and **when a complete hop is present** it
logs (sampled) and counts the learned edge (``this_uid -> next_hop_uid@peer_id`` and/or
``this_uid -> prev_hop_uid@peer_id``). It does NOT retain the hop and does NOT act on it -- no
dialing, no routing. Activations/grads still travel worker -> trainer -> worker exactly as today;
this is purely observational, proving the coordination is coherent. Acting on the hop (direct push)
is a later change.

It is the observational analogue of :class:`~agora_server.core.server.activation_cache.ActivationCache`
and mirrors that component's shape: constructed **once in the parent**
:class:`~agora_server.core.server.server.Server` (before the handler processes fork) and shared
across all of a worker's ``ConnectionHandler`` processes via an ``mp.Manager`` -- a microbatch's
forward lands on any one of them, so the counters cannot be process-local. Counters are surfaced
through a periodic ``[W2WHop]`` log line (scraped by ``WorkerPromMonitor``), the same in-process
mechanism :class:`ActivationCache` uses for ``[ActCache]``; no new metrics infrastructure.

Because a terminus (the tail has no next hop, the head no prev hop) and every legacy microbatch carry
no hop in that direction, "no hop present" is treated uniformly as nothing-to-learn (a silent no-op):
a terminus and a legacy batch are indistinguishable by design, so the learner never tries to tell them
apart and never logs a "terminus". ``learn`` is called on every forward AND backward RPC, so it is
cheap and never raises into the RPC path (any unexpected parse error is swallowed and counted).
"""

import multiprocessing as mp

from typing import Optional

from agora_server.core.batch_metadata import BatchMetadata
from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)

# Sample the per-event "next hop learned" log every Nth learned hop (mirrors the trainer's
# `[W2WRoute]` sampling cadence in swarm.utils.w2w_trainer_utils): structured logs + in-process
# counters are sufficient, and a worker sees a hop on every forward RPC, so
# logging each one would be far too noisy.
_LEARN_LOG_EVERY = 50


class NextHopLearner:
    """Observational next-hop learner shared by a worker's connection-handler processes.

    Construct **once in the parent process** (before the handlers fork) so all handlers inherit the
    same ``mp.Manager`` proxies. Stateless apart from the cross-process counters; it keeps no
    per-microbatch state and never acts on a learned hop.
    """

    def __init__(self, manager=None, *, name: str = ""):
        # Create the cross-process manager here (in the parent, before the handler processes fork)
        # when the caller does not supply one. Callers that own a manager (e.g. tests) may pass it in.
        self._owns_manager = manager is None
        self._manager = manager if manager is not None else mp.Manager()
        self._name = name
        self._counters = self._manager.dict()  # name -> int
        self._lock = self._manager.Lock()

    # -- public API ---------------------------------------------------------------------------

    def learn(self, worker_uid: str, metadata: bytes) -> None:
        """Observe (parse + log + count) the next and/or previous hop taught for one RPC.

        Bidirectional: one ``learn`` call handles either or both
        directions. The trainer's per-RPC metadata is disjoint -- a forward carries ``next_hop`` only
        (non-tail) or ``prev_hop`` only (fused tail), and a backward carries ``prev_hop`` only -- so
        counting each present pair once across every forward AND backward RPC never double-counts.
        For each direction:
        - complete hop (both ``*_hop_uid`` and ``*_hop_peer_id`` set) -> a "learned" event: a sampled
          ``[W2WHop]`` log line and ``next_hops_learned`` / ``prev_hops_learned`` bumped.
        - partial hop (exactly one of the two set -- the inconsistent signal that survives
          ``BatchMetadata``'s per-field type coercion) -> ``parse_errors`` bumped + a sampled warn.
        - no hop in that direction (legacy batch OR a terminus) -> silent no-op for it.

        Cheap, called on every forward AND backward RPC, and must never raise into the RPC path: any
        unexpected error is swallowed and counted as a parse error.
        """
        try:
            if not metadata:
                return  # legacy / no-metadata RPC: nothing to learn
            md = BatchMetadata.from_bytes(metadata)
            if md is None:
                return  # unparseable metadata is handled by the activation-cache path; not our signal
            next_uid, next_peer_id = md.next_hop_uid, md.next_hop_peer_id
            if next_uid and next_peer_id:
                self._on_next_learned(worker_uid, next_uid, next_peer_id)
            elif next_uid or next_peer_id:
                # Exactly one field set: an inconsistent/partial hop. from_bytes already coerced a
                # wrong-typed field to None, so this is the main malformed-hop signal that survives.
                self._on_parse_error(worker_uid, "next", next_uid, next_peer_id)
            # else: no next hop -> tail terminus or legacy batch -> silent no-op (indistinguishable).
            prev_uid, prev_peer_id = md.prev_hop_uid, md.prev_hop_peer_id
            if prev_uid and prev_peer_id:
                self._on_prev_learned(worker_uid, prev_uid, prev_peer_id)
            elif prev_uid or prev_peer_id:
                self._on_parse_error(worker_uid, "prev", prev_uid, prev_peer_id)
            # else: no prev hop -> head terminus or legacy batch -> silent no-op (indistinguishable).
        except Exception as e:
            # learn() must NEVER raise into the forward RPC path. The fallback accounting/logging is
            # itself best-effort: the mp.Manager proxy may be torn down during shutdown (touching it
            # raises BrokenPipeError/EOFError) and logging could fail too, so swallow everything here.
            err = type(e).__name__
            try:
                self._bump("parse_errors", 1)
                logger.warning(f"[W2WHop] learn failed for {worker_uid}: {err}")
            except Exception:
                pass

    def stats(self) -> dict:
        with self._lock:
            return {k: self._counters.get(k, 0) for k in _COUNTER_NAMES}

    def close(self) -> None:
        """Shut down the owned manager process (no-op if the manager was supplied by the caller)."""
        if self._owns_manager:
            self._manager.shutdown()

    # -- internals ----------------------------------------------------------------------------

    def _on_next_learned(self, worker_uid: str, next_hop_uid: str, next_hop_peer_id: str) -> None:
        n = self._bump("next_hops_learned", 1)
        if n % _LEARN_LOG_EVERY == 0:
            logger.info(f"[W2WHop] next hop learned: {worker_uid} -> {next_hop_uid}@{next_hop_peer_id}")

    def _on_prev_learned(self, worker_uid: str, prev_hop_uid: str, prev_hop_peer_id: str) -> None:
        n = self._bump("prev_hops_learned", 1)
        if n % _LEARN_LOG_EVERY == 0:
            logger.info(f"[W2WHop] prev hop learned: {worker_uid} -> {prev_hop_uid}@{prev_hop_peer_id}")

    def _on_parse_error(self, worker_uid: str, direction: str, uid: str | None, peer_id: str | None) -> None:
        n = self._bump("parse_errors", 1)
        if n % _LEARN_LOG_EVERY == 0:
            logger.warning(
                f"[W2WHop] partial {direction} hop for {worker_uid}: "
                f"uid={uid!r} peer_id={peer_id!r} (exactly one field set) -- ignored"
            )

    def _bump(self, name: str, delta: int) -> int:
        with self._lock:
            value = self._counters.get(name, 0) + delta
            self._counters[name] = value
            return value


_COUNTER_NAMES = (
    "next_hops_learned",
    "prev_hops_learned",
    "parse_errors",
)
