---
icon: material/book-open-page-variant
---

# Overview

Agora is a system for multi-party pipeline-parallel training. Many parties train one model together over WAN-grade links, and the parameters are partitioned across contributor GPUs so no one GPU has to hold the full model.

This setup forces a different design from a standard datacenter training stack. A contributor can disconnect mid-step. A consumer 4090 in one residential location may route activations to a datacenter A100 in another region. Over the lifetime of a long run the swarm sees hardware join and leave faster than nodes fail in a centralised cluster, and eventually it sees peers who join with the intent to disrupt. The sections below describe how each of these constraints shapes the design.

---

## Membership is volatile

Participants join and leave during a run. Total compute grows and shrinks; joiners arrive and depart far faster than centralised hardware would fail. Training therefore runs on a fluid set of devices, not a fixed block. Failure handling must be built into every collective primitive rather than added as a recovery path. [Fault Tolerance](fault-tolerance.md) describes the protocol-level mechanics.

---

## Bandwidth is residential

Co-locating peers is not an option, and there is no guarantee that two peers share a region. The realistic assumption is that any pair of peers communicates over the open internet.

!!! info "Bandwidth gap"
    Cloud datacenter training assumes 400 Gb/s – 3.2 Tb/s between nodes (NVLink / InfiniBand). Contributor WAN links run 200 Mbps – 1 Gbps: three to four orders of magnitude slower.

Closing that gap is the role of the [research foundations](research-foundations.md). SSN compresses the activations that travel between pipeline stages by up to 100×; async SPARTA replaces gradient AllReduce with a sparse 5% slice every 20 local steps. Together they keep per-step communication on each axis within the capacity of a residential link.

---

## Hardware is heterogeneous

Devices vary widely. An H100 might share a stage with a four-year-old A40; a residential 4090 has to coexist with a fibered datacenter rack. Compute heterogeneity is already an open problem on its own, and bandwidth and latency heterogeneity layer on top of it.

Heterogeneity is a hard requirement for Agora, not an optimisation, for three reasons.

- **Cost.** A 4090 sitting idle in a consumer system has zero marginal cost beyond electricity. There is no capex to amortise, no datacenter cooling bill, no on-call operator. The pool of such hardware is large.
- **Broad participation.** Many university departments have an old DGX and a handful of V100s. These contributors can participate, even at lower throughput.
- **Protocol resilience.** A model meant to be a public good has to be trainable without datacenter access. If a jurisdiction were to revoke cloud-GPU contracts on short notice, training should continue at reduced throughput rather than halt.

---

## Collectives must be peer-to-peer

NCCL and MPI assume one administrative domain owns every rank. That assumption fails the moment two peers belong to different parties: there is no administrator who can resynchronise the cluster after a failure, and any AllReduce participant may disappear before the round completes. Agora rebuilds the relevant primitives (AllReduce most prominently) so they are multi-party and tolerant of mid-collective failure.

The same rank assumption appears on the pipeline-parallel axis. In standard pipeline-parallel a batch flows through the stages of a pre-determined rank. Agora relaxes that requirement: stages must tolerate node failures gracefully and reroute batches peer-to-peer to whichever worker in the next stage is reachable, rather than to a fixed rank.

---

## Adversarial peers *(roadmap)*

!!! info "Future work: not yet implemented"
    An open, multi-party protocol attracts peers whose interest is in disrupting training rather than contributing to it. Defending against that is a known requirement on the protocol-resilience roadmap. It is **not yet implemented** in Agora today.

---

Together these constraints rule out the standard distributed-training stack. The rest of these pages describe what replaces it.
