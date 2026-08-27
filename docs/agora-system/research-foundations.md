---
icon: material/flask
---

# Research Foundations { .numbered }

Two compression schemes make residential-bandwidth training viable. **Subspace Networks** reduce the activations that travel between pipeline stages. **Async SPARTA** reduces the parameter sync that keeps same-stage replicas in agreement. They operate on different axes of the communication graph: SSN on the pipeline-parallel axis, SPARTA on the data-parallel axis.

---

## Subspace Networks

!!! info "Paper"
    [Protocol Models: Scaling Decentralized Training with Communication-Efficient Model Parallelism](https://arxiv.org/pdf/2506.01260)

Pipeline parallelism's cost is the bandwidth between adjacent stages. Forward activations flow downstream; activation gradients flow back along the same edges. Both are sized by the hidden state: batch × sequence × hidden. Uncompressed, this makes WAN-bandwidth training infeasible: a single sample's activation saturates a residential uplink for seconds, and the pipeline becomes communication-bound long before compute is.

Subspace Networks (SSN) integrate compression into the architecture itself. At every stage boundary, the hidden state is constrained to a low-rank subspace before crossing to the next stage, and the receiving stage reconstructs the full state from the subspace coefficients. The transformer block is modified so the forward-backward signal stays consistent through the projection: what the next stage receives, and what gradient flows back, is exactly what the model's bulk linear path expects. The compression is therefore lossless with respect to the backpropagated gradient signal, in contrast to lossy compression schemes that accumulate compression error across pipeline stages.

Two empirical observations make this work. First, the projection matrices of large pretrained transformers exhibit *rank collapse*: their effective rank is far below their nominal dimension, so constraining them to a shared learned low-rank subspace from the start costs little. Second, the residual stream is preserved at full rank where the architecture needs it; only the bulk linear path is run in the compressed subspace. The recursive structure of transformer blocks is reused so the same low-rank parameters can be shared across layers.

<figure class="agora-figure" markdown="0">
<div class="ssn-fig">
<svg viewBox="0 0 760 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Subspace Networks compression at a pipeline-stage boundary"><defs><marker id="ssn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#d73a49"/></marker><marker id="ssn-arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#bf8700"/></marker></defs><rect x="20" y="50" width="170" height="140" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/><text x="105" y="40" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" letter-spacing="1" fill="#999">STAGE N</text><text x="105" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="17" font-weight="700" fill="#111">Transformer block</text><text x="105" y="140" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#777">layers L₀ … Lₖ</text><rect x="190" y="90" width="36" height="80" rx="4" fill="#f0f7ff" stroke="#0969da"/><text x="208" y="115" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#0969da">SSN</text><text x="208" y="148" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#0969da">↓</text><line x1="226" y1="120" x2="552" y2="120" stroke="#d73a49" stroke-width="2" marker-end="url(#ssn-arrow)"/><text x="380" y="106" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" fill="#d73a49">compressed activation</text><text x="380" y="143" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#666">~ 5 Mbits / token batch</text><line x1="226" y1="155" x2="552" y2="155" stroke="#bf8700" stroke-width="2" marker-start="url(#ssn-arrow-gray)" stroke-dasharray="0"/><text x="380" y="178" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" fill="#bf8700">compressed activation gradient (backward)</text><rect x="552" y="90" width="36" height="80" rx="4" fill="#f0f7ff" stroke="#0969da"/><text x="570" y="115" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#0969da">SSN</text><text x="570" y="148" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#0969da">↑</text><rect x="588" y="50" width="170" height="140" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/><text x="673" y="40" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" letter-spacing="1" fill="#999">STAGE N+1</text><text x="673" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="17" font-weight="700" fill="#111">Transformer block</text><text x="673" y="140" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#777">layers Lₖ₊₁ … L₂ₖ</text><text x="380" y="222" text-anchor="middle" font-family="Inter,sans-serif" font-size="16" font-weight="600" fill="#333">Up to 100× less bandwidth than the uncompressed activation (~671 Mbits)</text></svg>
</div>
<figcaption>SSN constrains stage-boundary activations to a low-rank subspace; only the subspace coefficients cross the network, and the receiving stage reconstructs the full activation. The same path runs in reverse for activation gradients during the backward pass.</figcaption>
</figure>

A concrete example: a 8.5B-parameter transformer with a 5K embedding dimension, 32 layers, FP32 activations, and a 4K sequence length sends a 671 Mbit activation per stage boundary at batch size 1. Uncompressed that is several seconds on a 200 Mbps link, and the pipeline becomes communication-bound. SSN compresses the same activation to approximately 7 Mbits (up to 100× less), which keeps the boundary cost within typical step time.

---

## Asynchronous Sparse Parameter Averaging

!!! info "Paper"
    [AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism](https://arxiv.org/pdf/2601.22442)

Workers in the same stage are data-parallel replicas: they each train on different microbatches and should converge to the same parameters. The standard mechanisms for maintaining this agreement are gradient AllReduce and parameter AllReduce. Agora replaces gradient AllReduce entirely and uses a sparse variant of parameter AllReduce.

The first standard mechanism Agora supersedes is *gradient AllReduce*: every worker computes a local gradient, the cluster averages those gradients, and each worker applies the same update. This is the standard data-parallel training step (e.g. DDP). On a WAN link the all-reduce dominates step time, and one slow peer delays the rest of the stage. Async SPARTA forgoes gradient AllReduce: each worker computes its own gradient and runs its own local optimizer step. Replicas diverge as a result and require a separate mechanism to re-synchronize.

The second standard mechanism Agora supersedes is *parameter AllReduce*: periodically average the parameters themselves to remove the divergence. A full-parameter AllReduce on every cadence would still dominate the network link, so async SPARTA averages only a sparse subset of parameters per round. An averaging round runs once every 20 local steps and averages 5% of the parameter set across the matched group, with successive rounds covering non-overlapping slices until the full parameter set has been averaged. Communication scales with parameter count (the round transfers 5% × parameter_count × 4 bytes), but the constant fits within a residential uplink, and rounds run alongside ongoing local training so most of the averaging cost overlaps with compute.

Same-stage workers stay in close agreement without the training loop ever stopping for a synchronous all-reduce.
