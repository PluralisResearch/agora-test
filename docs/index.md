---
icon: material/home
---

# Pluralis-8B Collective Run

A decentralized pipeline-parallel pre-training run. 8B-parameter transformer, served by contributor GPUs over the public internet.

[Get started: Agora client](quick-start/index.md){ .md-button .md-button--primary }
&nbsp;
[Agora system overview](agora-system/overview.md){ .md-button }

---

## What is Pluralis-8B?

Pluralis-8B is a collective pre-training pilot on **Agora**, the system that connects a consumer GPU to a collaborative training run. Each participant hosts one pipeline stage of the model; participants can join or leave at any time, and adding more peers to a stage increases data-parallel throughput within that stage.

- **Consumer-grade hardware (minimum)**: 24 GB GPU (e.g. RTX 4090 or equivalent), 80 GB RAM, 80 GB disk, 200 Mbps network
- **Cross-platform**: Linux and Windows + WSL2 (CUDA)
- **One-command launch**: `python3 agora_cli.py`
- **Multi-GPU support**: run one node per GPU on the same machine
- **Live swarm participation**: join an ongoing run, synchronize state, then contribute compute and parameter updates

<figure class="agora-figure" markdown="0">
<div class="agora-hero">
  <div class="hero-zone">
    <div class="hero-label">Discovery</div>
    <div class="hero-title">Seeds</div>
    <div class="hero-desc">Stateless DHT bootstrap. Routing only, no model data.</div>
  </div>
  <div class="hero-arrow">→</div>
  <div class="hero-zone compute">
    <div class="hero-label">Compute</div>
    <div class="hero-title">Workers</div>
    <div class="hero-desc">Pipeline-parallel stages. Forward / backward / AllReduce. Heterogeneous GPUs.</div>
  </div>
  <div class="hero-arrow">↔</div>
  <div class="hero-zone">
    <div class="hero-label">Coordination</div>
    <div class="hero-title">Trainers</div>
    <div class="hero-desc">CPU only. Microbatch routing, load balancing, data sharding.</div>
  </div>
</div>
<figcaption>Three roles, three layers. 
<a href="agora-system/training-architecture/">See the full Training Architecture for the per-stage view.</a></figcaption>
</figure>

---

## Earning points

Every node accrues a **score** combining the raw pflops it processes with a baseline 1 PFLOP per hour for time spent active in the swarm. The dashboard sums scores across all the peers running under one account and ranks contributors live on the [public leaderboard](https://dashboard.pluralis.ai/).

!!! info "Higher = more pflops"
    More uptime and faster GPUs both translate directly to a higher rank.

[Read the full Points & Leaderboard guide](quick-start/points.md){ .md-button .md-button--primary }

---

## Research

Four published works underwrite the design of Agora.

1. **Protocol Models: Scaling Decentralized Training with Communication-Efficient Model Parallelism**.&nbsp;[arXiv:2506.01260](https://arxiv.org/pdf/2506.01260). Subspace Networks (SSN), the architectural compressor that reduces the activation crossing each pipeline-stage boundary by up to 100×. The mechanism that makes WAN-grade pipeline parallelism viable.
2. **AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism**.&nbsp;[arXiv:2601.22442](https://arxiv.org/pdf/2601.22442). Agora uses the asynchronous sparse parameter averaging from this paper: same-stage workers AllReduce 5% of their parameters every 20 local steps, with successive rounds covering non-overlapping slices, in parallel with ongoing training. Data-parallel synchronization never stalls the training loop on a full all-reduce.
3. **Pluralis' Multi-party Training Stack**.&nbsp;[pluralis.ai/blog](https://pluralis.ai/blog/pluralis-multi-party-training-stack/). The engineering write-up that integrates the individual mechanisms into a complete system.
4. **SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient**.&nbsp;[arXiv:2301.11913](https://arxiv.org/pdf/2301.11913). The original distributed-pipeline paper Agora builds on.