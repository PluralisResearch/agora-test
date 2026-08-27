---
icon: octicons/cpu-16
---

# Training Architecture { .numbered }

Three layers compose an Agora training run. *Discovery* runs a stateless DHT for peer-to-peer addressing. *Compute* is the pipeline itself: workers, each holding one stage's parameters, running forward and backward. *Coordination* is a small set of CPU-only trainers that route microbatches through the pipeline and shard the dataset across them. The full system runs on heterogeneous, untrusted hardware without any one contributor holding the complete model weights.

<figure class="agora-figure" markdown="0">
<div class="swarm-fig">
  <div class="architecture">
    <div class="zone seed-zone">
      <div class="zone-header">
        <div class="zone-number">Discovery Layer</div>
        <div class="zone-title">Seeds</div>
        <div class="zone-desc">DHT bootstrap nodes. Peer registration &amp; discovery.</div>
      </div>
      <div class="zone-body">
        <div class="seed-item">
          <div class="seed-name">Seed 0</div>
          <div class="seed-desc">Primary DHT bootstrap. Publishes multiaddr for initial peer connections.</div>
        </div>
        <div class="seed-item">
          <div class="seed-name">Seed 1</div>
          <div class="seed-desc">Redundant bootstrap. Ensures discovery continues if Seed 0 is unreachable.</div>
        </div>
      </div>
      <div class="zone-footer">
        <div class="security-badge">
          <div class="badge-dot"></div>
          <div class="badge-text"><strong>Stateless.</strong> Routing only, no model data.</div>
        </div>
      </div>
    </div>
    <div class="arrow-col">
      <div class="data-flow">
        <div class="flow-direction">Peer Discovery</div>
        <div class="flow-arrow-line flow-discovery"><div class="line"></div><div class="head-right"></div></div>
        <div class="flow-payload">
          <div class="payload-what">Multiaddr exchange</div>
          <div class="payload-size">libp2p / DHT</div>
        </div>
      </div>
    </div>
    <div class="zone pipeline-zone">
      <div class="zone-header">
        <div class="zone-number">Compute Layer</div>
        <div class="zone-title">Training Pipeline</div>
        <div class="zone-desc">Workers hold one pipeline stage's parameters, process fwd/bwd, run periodic SPARTA state averaging within their stage.</div>
      </div>
      <div class="zone-body">
        <div class="stage-grid">
          <div class="pipeline-stage head">
            <div class="stage-label">Stage 0</div>
            <div class="stage-name">Head</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">H100 80G</div><div class="worker-info">6 layers + embed</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">H100 80G</div><div class="worker-info">6 layers + embed</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage">
            <div class="stage-label">Stage 1</div>
            <div class="stage-name">Body 1</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">RTX 4090</div><div class="worker-info">4 layers</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">RTX 4090</div><div class="worker-info">4 layers</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage ellipsis">
            <div class="stage-label">…</div>
            <div class="stage-name">Bodies 2–4</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">mixed</div><div class="worker-info">4 layers each</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">mixed</div><div class="worker-info">4 layers each</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage">
            <div class="stage-label">Stage 5</div>
            <div class="stage-name">Body 5</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">L40S 48G</div><div class="worker-info">4 layers</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">L40S 48G</div><div class="worker-info">4 layers</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage tail">
            <div class="stage-label">Stage 6</div>
            <div class="stage-name">Tail</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">A100 40G</div><div class="worker-info">6 layers + lm_head</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">A100 40G</div><div class="worker-info">6 layers + lm_head</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
        </div>
      </div>
      <div class="zone-footer">
        <div class="security-badge">
          <div class="badge-dot"></div>
          <div class="badge-text"><strong>Model split across stages</strong> &mdash; each worker holds one stage's parameters, not the full model.</div>
        </div>
      </div>
    </div>
    <div class="arrow-col">
      <div class="data-flow">
        <div class="flow-direction">Activations &rarr;</div>
        <div class="flow-arrow-line flow-forward"><div class="line"></div><div class="head-right"></div></div>
        <div class="flow-payload"><div class="payload-what">Forward pass</div><div class="payload-size">Head &rarr; Body &rarr; Tail</div></div>
      </div>
      <div class="data-flow">
        <div class="flow-direction">&larr; Activation gradients</div>
        <div class="flow-arrow-line flow-backward"><div class="line"></div><div class="head-left"></div></div>
        <div class="flow-payload"><div class="payload-what">Backward pass</div><div class="payload-size">Tail &rarr; Body &rarr; Head (<code>grad_input</code> only)</div></div>
      </div>
    </div>
    <div class="zone trainer-zone">
      <div class="zone-header">
        <div class="zone-number">Coordination Layer</div>
        <div class="zone-title">Trainers</div>
        <div class="zone-desc">Microbatch coordination, load balancing, data management.</div>
      </div>
      <div class="zone-body">
        <div class="trainer-item">
          <div class="trainer-name">Trainer 0</div>
          <div class="trainer-desc">Microbatch coordination<br>Load balancing<br>Data management</div>
          <div class="trainer-tag tag-cpu">CPU only</div>
        </div>
        <div class="trainer-item">
          <div class="trainer-name">Trainer 1</div>
          <div class="trainer-desc">Microbatch coordination<br>Load balancing<br>Data management</div>
          <div class="trainer-tag tag-cpu">CPU only</div>
        </div>
      </div>
      <div class="zone-footer">
        <div class="security-badge">
          <div class="badge-dot"></div>
          <div class="badge-text"><strong>Pluralis-owned only</strong> &mdash; not on contributor nodes.</div>
        </div>
      </div>
    </div>
  </div>
</div>
<figcaption>Three-zone Agora architecture: Discovery (Seeds), Compute (Workers), Coordination (Trainers). Heterogeneous example configuration; real swarms vary by participant hardware. <span class="fig-forward">Forward</span> activations flow Head → Body → Tail; <span class="fig-backward">backward</span> activation gradients (<code>grad_input</code>) flow Tail → Body → Head. Parameter gradients never cross stage boundaries.</figcaption>
</figure>

## Discovery Layer { .t-discovery }

A seed is a stateless <span class="t-discovery">DHT</span> bootstrap. Workers query a seed for an entry point into the swarm and then communicate with other peers directly; the seed never stores model data of its own. Pluralis runs two seeds for redundancy: if Seed 0 is unreachable, Seed 1 serves the same role, and a new worker can bootstrap from either one.

## Compute Layer { .t-compute }

A worker holds the model parameters and performs the compute. It owns one pipeline stage (Head, Body, or Tail) and runs forward and backward passes for batches the trainer routes to it. Workers within the same stage participate together in periodic, async SPARTA averaging rounds. The [Workers section](#workers) below covers the runtime structure.

## Coordination Layer { .t-coord }

A trainer holds no model parameters. Its role is to orchestrate the pipeline: route microbatches to a healthy worker in each stage, balance load across the workers in each stage, and supply dataset shards. Trainers run on CPU and run on Pluralis-owned infrastructure rather than on contributor nodes; see the [Trainers section](#trainers) for fault-tolerance and load-balancing details.

---

## Component Deep Dive

### Workers

A worker is a single process holding one stage's parameters (one or more transformer layers). It performs forward and backward on those parameters, runs its own local optimizer to apply gradients, and joins same-stage peers in periodic `AllReduce` rounds for state averaging. A worker has no knowledge of the rest of the pipeline: only its own stage.

<figure class="agora-figure" markdown="0">
<div class="worker-fig">
<div class="worker-label">Worker: Stage X</div>
<div class="worker-grid">
<div class="wc-cell"><div class="wc-name">Connection Handlers</div><div class="wc-desc">Listen for trainer gRPC requests; place batches in fwd / bwd queues. Multiplexed on the same port.</div></div>
<div class="wc-cell accent"><div class="wc-name">Runtime</div><div class="wc-desc">Loops over fwd / bwd queues and dispatches batches into ModuleBackend for execution.</div></div>
<div class="wc-cell"><div class="wc-name">ModuleBackend</div><div class="wc-desc">Stores the nn.Module for this stage. Owns the forward / backward task pools.</div></div>
<div class="wc-cell"><div class="wc-name">DHTHandler</div><div class="wc-desc">Declares this Worker's availability in its stage (head.0.0, body1.0.1, …) for trainer + peer discovery.</div></div>
<div class="wc-cell accent"><div class="wc-name">SPARTA Optimizer</div><div class="wc-desc">Accumulates gradients locally, runs the local optimizer step, then matches with same-stage peers and AllReduces 5% of parameters.</div></div>
<div class="wc-cell"><div class="wc-name"><span class="t-discovery">DHT</span></div><div class="wc-desc">Hivemind Kademlia DHT. Peer discovery, expert registration, progress tracking, matchmaking.</div></div>
</div>
</div>
<figcaption>Worker internals: six co-resident components inside a single Worker process. <strong>Runtime</strong> drives <strong>ModuleBackend</strong> for compute; the <strong>SPARTA Optimizer</strong> coordinates the parameter-averaging step with same-stage peers via the shared <strong>DHT</strong>.</figcaption>
</figure>

#### DHT

Agora uses Hivemind's Kademlia DHT for four functions: peer discovery, expert registration, progress tracking, and matchmaking for `AllReduce`.

#### ModuleBackend

The `nn.Module` for this stage and the forward and backward functions the `Runtime` invokes. Also owns the two task pools (forward and backward) where incoming trainer requests queue up before the `Runtime` processes them.

#### Async SPARTA

Each worker accumulates gradients from its own backward passes and runs its own optimizer step locally; there is no per-step gradient AllReduce. Same-stage replicas drift apart as a result. To re-synchronize, every 20 local steps the worker matches with same-stage peers over the DHT and AllReduces 5% of its parameters. Successive rounds cover non-overlapping slices, so the full parameter set has cycled through over a 20-round window.

!!! info "Paper"
    [AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism](https://arxiv.org/pdf/2601.22442)

#### Connection Handlers

gRPC listeners that receive trainer requests and put each batch into the right queue: forward or backward. Multiple listeners share a single port.

#### DHTHandler

A background thread that keeps re-announcing this worker in the DHT under its stage-prefixed UID (`head.0.0`, `body1.0.1`, `tail.0.0`). Trainers read the announcements to find workers; same-stage peers use them for matchmaking during `AllReduce`.

#### Runtime

The main loop. Dequeues batches from the forward and backward queues and runs them through `ModuleBackend`. On the backward path it rebuilds the autograd graph by re-running the forward (see [activation recomputation](memory-communication.md#activation-recomputation) for details), calls `torch.autograd.backward()`, and triggers the optimizer step at the appropriate point in the batch-size accumulator.

---

#### Batch processing

Once running, the Worker runs an event loop processing batches from trainers:

1. Trainer sends a forward request via gRPC → Connection Handler places it in the forward queue.
2. Runtime dequeues the batch → calls `ModuleBackend.forward()` → returns the output.
3. Trainer sends a backward request with gradient outputs → Connection Handler places it in the backward queue.
4. Runtime dequeues the batch → calls `ModuleBackend.backward()` → triggers the optimizer step.

#### See also

- **How a new Worker joins a running swarm** (state download, queue, sync mode) → [Contributor Join Flow](startup-sequence.md).
- **What happens at the optimizer step** (ProgressTracker, Matchmaking, SPARTA AllReduce) → [Communication Patterns](memory-communication.md#communication-patterns).
- **Sync-mode entry / exit conditions and Worker-failure handling** → [Fault Tolerance](fault-tolerance.md).

---

### Trainers

The trainer's role looks like an ordinary PyTorch training loop: forward through the model, compute a loss, call backward. The difference is that none of those calls run locally. Every `forward` and `backward` is dispatched over the network to a worker holding the relevant stage. The trainer's responsibilities are to track the full pipeline topology, select a healthy worker for each stage, and route activations forward and activation gradients back. Parameter gradients themselves never leave a worker, and the trainer never holds parameters of its own.

<figure class="agora-figure" markdown="0">
<div class="trainer-fig">
<svg viewBox="0 0 760 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Trainer pipeline diagram: Trainer orchestrating remote workers across head / body / tail stages">
  <defs>
    <marker id="tr-arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#d73a49"/></marker>
    <marker id="tr-arr-warn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#bf8700"/></marker>
  </defs>

  <!-- Trainer node at top centre -->
  <rect x="280" y="20" width="200" height="60" rx="6" fill="#ffffff" stroke="#999" stroke-width="1.5"/>
  <text x="380" y="44" text-anchor="middle" font-family="Inter,sans-serif" font-size="15" font-weight="700" fill="#111">Trainer</text>
  <text x="380" y="62" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#555">CPU · holds no parameters</text>

  <!-- Trainer → workers: stems -->
  <line x1="380" y1="80" x2="380" y2="115" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="392" y="100" font-family="Inter,sans-serif" font-size="11" fill="#444">libp2p gRPC · forward / backward</text>

  <!-- Connector branching to four stages -->
  <line x1="100" y1="115" x2="660" y2="115" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="100" y1="115" x2="100" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="287" y1="115" x2="287" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="473" y1="115" x2="473" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="660" y1="115" x2="660" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="3 3"/>

  <!-- Stage boxes -->
  <g font-family="Inter,sans-serif">
    <rect x="40" y="150" width="120" height="120" rx="6" fill="#f0f7ff" stroke="#0969da" stroke-width="1.5"/>
    <text x="100" y="174" text-anchor="middle" font-size="11" font-weight="600" letter-spacing="1" fill="#0969da">STAGE 0</text>
    <text x="100" y="196" text-anchor="middle" font-size="14" font-weight="700" fill="#111">Head</text>
    <text x="100" y="220" text-anchor="middle" font-size="11" fill="#444">embed · early layers</text>
    <text x="100" y="246" text-anchor="middle" font-size="10" fill="#777">Pipe 0 / Pipe 1</text>

    <rect x="227" y="150" width="120" height="120" rx="6" fill="#f9f9f9" stroke="#0969da" stroke-width="1.5"/>
    <text x="287" y="174" text-anchor="middle" font-size="11" font-weight="600" letter-spacing="1" fill="#777">STAGE 1</text>
    <text x="287" y="196" text-anchor="middle" font-size="14" font-weight="700" fill="#111">Body 1</text>
    <text x="287" y="220" text-anchor="middle" font-size="11" fill="#444">transformer block</text>
    <text x="287" y="246" text-anchor="middle" font-size="10" fill="#777">Pipe 0 / Pipe 1</text>

    <rect x="413" y="150" width="120" height="120" rx="6" fill="#f9f9f9" stroke="#0969da" stroke-width="1.5"/>
    <text x="473" y="174" text-anchor="middle" font-size="11" font-weight="600" letter-spacing="1" fill="#777">STAGE 2</text>
    <text x="473" y="196" text-anchor="middle" font-size="14" font-weight="700" fill="#111">Body 2</text>
    <text x="473" y="220" text-anchor="middle" font-size="11" fill="#444">transformer block</text>
    <text x="473" y="246" text-anchor="middle" font-size="10" fill="#777">Pipe 0 / Pipe 1</text>

    <rect x="600" y="150" width="120" height="120" rx="6" fill="#f0f7ff" stroke="#0969da" stroke-width="1.5"/>
    <text x="660" y="174" text-anchor="middle" font-size="11" font-weight="600" letter-spacing="1" fill="#0969da">STAGE 3</text>
    <text x="660" y="196" text-anchor="middle" font-size="14" font-weight="700" fill="#111">Tail</text>
    <text x="660" y="220" text-anchor="middle" font-size="11" fill="#444">final layers · loss</text>
    <text x="660" y="246" text-anchor="middle" font-size="10" fill="#777">Pipe 0 / Pipe 1</text>
  </g>

  <!-- Forward direction band (red, matching .fig-forward in figcaption) -->
  <line x1="160" y1="290" x2="600" y2="290" stroke="#d73a49" stroke-width="2" marker-end="url(#tr-arr-blue)"/>
  <text x="380" y="284" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="600" fill="#d73a49">Forward: activations · Head → Body → Tail</text>

  <!-- Backward direction band -->
  <line x1="600" y1="310" x2="160" y2="310" stroke="#bf8700" stroke-width="2" marker-end="url(#tr-arr-warn)"/>
  <text x="380" y="304" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="600" fill="#bf8700">Backward: grad_input · Tail → Body → Head</text>
</svg>
</div>
<figcaption>The Trainer (CPU, no parameters) orchestrates remote workers across pipeline stages over libp2p gRPC. <span class="fig-forward">Forward</span> activations flow Head → Body → Tail; <span class="fig-backward">backward</span> activation gradients flow Tail → Body → Head. Two pipes per stage give data-parallel redundancy.</figcaption>
</figure>

#### Training flow

##### Startup

1. Loads configuration and tokenizer config.
2. Prepares the dataset from Hugging Face.
3. Creates DHT connections using seed peers. Each model stage has its own dedicated DHT, enabling partitioning across stages.
4. Using the DHT, the Trainer discovers workers in each stage; all stages must show at least one available Worker before training can start.

##### Training loop

For each batch, the trainer iterates through the pipeline in order:

```python
hidden = head.forward(input_ids[:, :-1])
hidden = body1.forward(hidden)
hidden = body2.forward(hidden)
loss   = tail.forward(hidden, input_ids[:, 1:])   # shifted labels for LM
loss.backward()                                   # triggers backward on all stages
```

Worker selection within a stage uses a min-heap keyed by accumulated virtual runtime: the least-loaded worker is selected. When a worker finishes a request, its runtime is credited with the task's estimated duration and the worker re-enters the heap; new arrivals enter at the end. A transient network error triggers a retry against the same worker or a different one depending on the error class, and an unreachable worker is **short-banned** from the heap for 30s. A pipeline step only stalls if a stage empties entirely: every worker in that stage unreachable or short-banned at once.

Each pipeline stage gets its own <span class="t-discovery">DHT</span> connection on the trainer, so worker discovery for stage `head` is independent of discovery for stage `body3`. The full protocol for how workers and trainers announce and refresh each other is in [Communication Patterns → Periodic node announcement](memory-communication.md#periodic-node-announcement-peer-discovery).
