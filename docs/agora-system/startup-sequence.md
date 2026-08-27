---
icon: material/play-circle
---

# Contributor Join Flow

When a contributor's client starts, the swarm it is joining is already mid-run: peers are actively forwarding and backwarding microbatches, accumulating progress toward the next optimizer step, and running SPARTA averaging rounds. The new peer has to integrate without disrupting it. The sections below trace what happens from `python3 agora_cli.py` to the first batch this peer contributes.

---

## 1. Authorization

The first action the client performs is exchanging the contributor's HuggingFace token for a contributor slot. The auth service uses the token for identity only; no read/write scopes are required. A contributor **identity** is bound to the token. Any other peer started under the same token aggregates against the same identity on the leaderboard, so a multi-GPU contributor accumulates a single combined score. If the swarm is at capacity, the join is rejected at this step. If there is still room but other recent joins are still being integrated, the client waits briefly in the join queue; see [Queue](#3-queue). On success, a single log line confirms the slot and the assigned stage.

---

## 2. State download

The worker has no parameters at this point. The auth service provides the client with a checkpoint URL pointing to S3 (a periodic snapshot of the assigned stage), and the client downloads the parameters and optimizer state directly from there over HTTPS. For Pluralis-8B at ~900M parameters per stage, this takes tens of seconds to a few minutes on a stable 200 Mbps link. This is the most bandwidth-sensitive step in the entire join flow; an unstable link manifests here before it manifests anywhere else.

The S3 snapshot lags the live parameter state; it is published periodically, not on every step. Synchronizing to the swarm's current training state is the role of the sync phases described below.

---

## 3. Queue

The queue is not an at-capacity waiting list: if the swarm is already full the join is rejected outright at authorization, with no waiting list. The queue paces incoming joins when there is still room: it prevents many simultaneous joins from overwhelming trainers and gives the swarm time to stabilize after each new peer joins. The client remains queued with the downloaded state retained, so when its position clears it transitions directly to sync mode without re-downloading. The client logs a position estimate once per minute, and live per-stage occupancy is on the [Dashboard](https://dashboard.pluralis.ai/) under **Overview**. If concurrent-join pacing is not required at authorization time, this step is skipped entirely.

---

## 4. Sync Phase 1: weight synchronisation

The node is now formally in the swarm but invisible to trainers. Same-stage peers send the node the averaged parameters during each SPARTA round; the node's own weights are not yet incorporated into the average, and no batches are routed to it. The CLI prints:

```
[SYNC] Synchronising weights with peers. Node won't process batches in this phase.
[SYNC] This phase will last <N> steps (until local epoch <E>).
```

- **DHT record**: `sync_phase=1`, which trainers filter on to skip this peer.
- **Averaging participation**: `weight=0`. The node reads the averaged parameters every round but contributes nothing back to the average.
- **Duration**: typically several hours (`sync_phase1_steps`, currently 400 steps, ~20 SPARTA rounds).

This allows the local weights to converge toward synchronization with the rest of the swarm before any updates begin to affect the average. **No compute points accrue during Phase 1** since the node is not processing batches yet, but presence points start crediting from this phase onward, since the peer is now reporting to DHT.

---

## 5. Sync Phase 2: optimizer warm-up

Trainers now see the node. Real batches are routed to it, and the node processes them like any other peer. The node's weights still do not enter the SPARTA average. CLI:

```
[SYNC] Synchronising optimizer state. Node is now processing batches, but doesn't contribute to weight averaging yet.
[SYNC] This phase will last <N> steps (until local epoch <E>).
```

- **DHT record**: `sync_phase=2`. Trainers route batches to the node.
- **Local optimizer**: accumulates state and takes local steps as normal.
- **Averaging participation**: still `weight=0`. The node receives but does not contribute.
- **Progress counter**: samples processed in this phase do **not** count toward the per-stage `target_batch_size`; the node reports 0 to the progress tracker until sync exits.
- **Duration**: `sync_phase2_steps` (currently 100 steps).

The purpose of Phase 2 is to warm up the local optimizer state (AdamW moments and gradient norms) so that when the local weights begin contributing to averaging, their update direction is in agreement with the rest of the swarm. Because the node is processing real batches in this phase, its FLOPs are accumulated by the metrics server, and **compute points start accruing here** (presence points have been accruing since Phase 1).

---

## 6. Active

After Phase 2 exits, the node is a fully participating peer. CLI:

```
[SYNC] Sync complete. Node is now fully contributing to training.
```

- **DHT record**: `sync_phase=OFF`.
- **Averaging participation**: the node's weights enter the SPARTA average at full weight.
- **Progress counter**: the node's samples count toward the per-stage `target_batch_size`. Points continue to accrue as in Phase 2.

From here on the node processes batches indefinitely. If it disconnects, [Fault Tolerance](fault-tolerance.md) describes how the swarm handles the lost contribution and how a worker re-enters sync mode on rejoin if it has lagged too far behind.

---

## Summary

| Step | DHT `sync_phase` | Visible to trainers? | Contributes to averaging? | Notes |
|---|---|---|---|---|
| 1. Authorization | — | — | — | Token validated, stage assigned; rejected if swarm is full |
| 2. State download | — | No | No | Stale snapshot downloaded from S3 |
| 3. Queue (if pacing needed) | — | No | No | Brief wait while concurrent joins are integrated |
| 4. Sync Phase 1: weights | `1` | No | Receives only (`weight=0`) | Synchronize to live state; presence points only |
| 5. Sync Phase 2: optimizer | `2` | Yes | Receives only (`weight=0`) | Warm up local optimizer; compute points start accruing |
| 6. Active | `OFF` | Yes | Full weight | Steady-state contribution; compute and presence points accrue |
