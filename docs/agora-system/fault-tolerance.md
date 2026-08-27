---
icon: material/shield-half-full
---

# Fault Tolerance

Peers frequently disconnect, fail, or experience transient connection issues. Agora is built to be fault-tolerant and robust to these failure modes; the run is not stalled during crucial all-reduce operations, and valid work from healthy peers is preserved.

---

## Peer Failure During AllReduce { .t-failure }

An all-reduce round must tolerate slow or disconnecting peers partway through. The system divides tensors into smaller chunks and enforces a strict time limit for reducers to receive these chunks. Any failure in a peer sending its chunks (disconnect, bad connection) results in the sender being banned for the round, allowing the all-reduce to complete. The chunking preserves partial all-reduce results, so valid work from peers is not discarded. A reciprocal timeout mechanism protects the return path, where reducers send the reduced tensor chunks back. A full-round timeout protects against additional stalls from failed, disconnected, or slow peers; the round retains its partial all-reduce result.

There is no retry: failures are logged and training continues. Subsequent matchmaking and all-reduce operations run independently of previous failures, with the caveat that contributors that consistently fail are removed from the swarm.

!!! info "Key point"
    A peer failure is fatal only if an entire pipeline stage empties. As long as at least one worker remains in each stage, the trainer can route batches and training continues.

---

## Stale-state re-integration (sync mode) { .t-failure }

Sync mode is the mechanism by which a worker that is *behind* the swarm (re)joins it without biasing the live model toward its stale weights. The contributor-facing description is in [Contributor Join Flow](startup-sequence.md#4-sync-phase-1-weight-synchronisation); the fault-handling rule is:

A worker enters sync mode whenever it has lagged more than `max_allowed_stale` training steps behind the swarm, which happens in three ways: it lagged during training, the swarm progressed past it while it was downloading initial state, or it restored from a stale checkpoint.

In **Phase 1**, the worker is invisible to trainers (they parse the suffix and skip it) and joins SPARTA averaging rounds with `weight=0`, receiving the averaged parameters but contributing nothing.

In **Phase 2**, the worker becomes visible to trainers and starts processing batches, but still averages with `weight=0`. This is the optimizer-warmup phase: local optimizer state converges while the swarm's averaged parameters continue moving the worker's weights toward the live state.

Once a worker exists the synchronization mode, it transitions to full participation: its weights contribute to averaging and its samples count toward step progress.

---

## Join Blocking Window

The third failure mode is a peer joining mid-`AllReduce`: its state download would read parameters that are partway through being updated, and the result would be neither the pre-round nor the post-round model. To prevent that, new joins are blocked during the steps immediately around an averaging round. A joining peer can only download state during a stable window when no averaging round is in progress.
