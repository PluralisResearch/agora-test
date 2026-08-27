---
icon: material/database
---

# Memory & Communication { .numbered }

This page covers two questions: which tensors reside on which hardware during training, and which of them cross the network for the swarm to make progress.

---

## What Lives Where

Under the default config (`offload_optimizer: true`, `use_mixed_precision: true`) the Adam moments are offloaded to host RAM and the forward pass runs in BF16. The full layout:

| Tensor | Dtype (mixed_precision=true) | Dtype (mixed_precision=false) | Location (default) | Location (offload=false) |
| --- | --- | --- | --- | --- |
| Model parameters | FP32 | FP32 | GPU | GPU |
| Param gradients (`param.grad`) | FP32 | FP32 | GPU | GPU |
| Adam exp\_avg | FP32 | FP32 | CPU | GPU |
| Adam exp\_avg\_sq | FP32 | FP32 | CPU | GPU |
| Saved activations | BF16 | FP32 | GPU | GPU |
| Forward / backward matmuls | BF16 (tensor cores) | FP32 | GPU | GPU |
| AllReduce buffers | FP32 | FP32 | CPU | CPU |

`use_mixed_precision: true` wraps the forward pass in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`, so matmuls run on BF16 tensor cores and saved activations are stored in BF16. Each backward op runs in the dtype its matching forward op used. Parameters and `param.grad` remain FP32 throughout: PyTorch enforces matching dtypes between a parameter and its gradient.

`GradScaler` (the mechanism mixed precision normally uses to prevent FP16 underflow) is unnecessary here. BF16 has the same 8-bit exponent as FP32, so its dynamic range is identical and gradient underflow is not a risk.

Native BF16 tensor cores are required (compute capability ≥ 8.0): A100, H100, L40S, RTX 3090/4090, and similar. T4 and V100 do not have them; because they would emulate BF16 and run slower than FP32, those cards are excluded.

---

## Activation Recomputation

When a backward pass reaches a worker, the worker re-runs its own forward pass before calling `torch.autograd.backward()`. This is unusual and easy to mistake for gradient checkpointing, but the two mechanisms differ; this section explains why.

PyTorch's autograd graph is process-local. The graph holds references to in-process Python objects, intermediate tensors, and the grad metadata attached to them, none of which is serialisable across a network. In a pipeline-parallel run that spans multiple processes (typically multiple machines), the activations leaving one worker and arriving at the next stage cannot carry their autograd graph with them.

To send activations between stages over libp2p gRPC, Agora *detaches* them at every pipeline-stage boundary. The receiving worker sees a fresh leaf tensor with no history of how it was produced.

That detachment leaves a worker's local autograd graph severed at the input. When the backward pass arrives carrying a gradient with respect to that stage's output, there is no graph for autograd to traverse, and calling `.backward()` directly would do nothing.

The fix is to rebuild the graph on demand. At the start of the backward call, the worker re-runs the forward pass on its saved input, which constructs a fresh autograd graph local to this process. Backward then traverses that graph normally, populating `param.grad` for the worker's parameters.

!!! info "Not gradient checkpointing"
    Gradient checkpointing (`torch.utils.checkpoint`) is a *memory* optimisation: skip saving activations during forward, recompute them lazily during backward, trade ~33% extra compute for ~60-80% less activation memory. Agora's recomputation looks superficially similar but is forced by the **cross-process pipeline boundary**, not chosen for memory. Activations are fully materialised in this worker's memory either way.

---

## Memory Footprint

Workers fit on a 24 GB consumer GPU under the default config; see [Requirements](../quick-start/requirements.md) for hardware bounds. Host RAM holds the offloaded optimizer state, which is why the requirements call for 80 GB system RAM per GPU.

---

## Communication Patterns

A training run has five distinct types of network traffic. Discovery is slow and periodic. Activation traffic is per-microbatch. Batch-size accumulation and matchmaking use the DHT with minimal overhead. State averaging is rare, heavy, and synchronised between same-stage peers. Each is described in turn below.

### Periodic node announcement (peer discovery)

Workers and trainers re-announce themselves into the DHT on a timer. A worker publishes under its stage-prefixed UID (`head.0.0`, `body1.0.1`, `tail.0.0`), tagged with its current `sync_phase`. The `DHTHandler` thread refreshes every ~30s.

Trainers traverse each stage's DHT to maintain a live view of the active peers there, performing a full refresh of the per-stage roster every 600s. There is no central registry and no leader election: a peer joining or leaving is a write to the DHT that other peers observe on their next refresh.

### Forward / backward activation traffic

Per microbatch, the trainer issues an `rpc_forward` to a worker in each stage, in pipeline order:

- **Request payload**: input tokens (head) or hidden states from the previous stage (body / tail), plus the optional `grad_output` on backward calls.
- **Response payload**: hidden states (head / body), the loss tensor (tail's forward), or `grad_input` (any stage's backward).

Activation gradients (`grad_input`) flow back stage-by-stage during the backward chain. Parameter gradients never traverse the network: they are accumulated locally and consumed by each worker's own `optimizer.step()`. The only cross-worker averaging is the sparse **state** AllReduce described below. See [Activation Recomputation](#activation-recomputation) for how the backward chain works.

### Batch-size accumulation sync (`ProgressTracker`)

A worker cannot decide on its own when to step the optimizer; that decision is collective per stage. Each worker publishes its running sample count to the DHT via `ProgressTracker`. When the per-stage sum crosses `target_batch_size`, every worker in the stage takes its optimizer step in lockstep.

### Matchmaking (pre-cursor to SPARTA averaging)

Before a SPARTA round runs, same-stage peers agree on a group: who is participating, who is the round leader, what the round number is. Hivemind's matchmaking protocol does this over the DHT:

- Peers register their willingness to average this round.
- A group leader is elected (lowest peer ID among the candidates).
- Group composition is locked in and broadcast back.

Matchmaking runs once every 20 local optimizer steps.

### SPARTA state averaging (AllReduce within the matched group)

Once a group is formed, its members run a butterfly `AllReduce` on a sparse parameter subset:

- **Sparse**: 5% of the parameter set is averaged per round.
- **Rotating partitions**: a different 5% slice each round, covering the full parameter set in 20 rounds.
- **Uncompressed**: parameter slices traverse the network in FP32 (4 bytes per parameter).

**Optimizer state is not periodically averaged.** AdamW's `exp_avg` and `exp_avg_sq` stay local during training; they are only transferred during the initial state download when a worker joins the run.

#### Bandwidth per averaging round

For a body stage of the Pluralis-8B (~907M trainable parameters), each round transmits 5% × 907M × 4 bytes ≈ **181 MB** per peer. The head and tail are larger and asymmetric. The head carries six layers plus a low-rank reparametrized token embedding (~1.37B trainable, ~274 MB per round). The tail is by far the largest stage at ~2.73B trainable parameters (~546 MB per round), carrying six layers plus a full-rank output projection, and its transformer layers use uncompressed projections rather than the low-rank reparametrizations used in the head and bodies. Spread across the group's butterfly all-reduce, this still fits within the WAN bandwidth budget.
