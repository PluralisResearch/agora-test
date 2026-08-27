---
icon: material/play-circle
---

# Running Agora

## Start the node

```bash
git clone https://github.com/PluralisResearch/agora
cd agora
python3 agora_cli.py
```

On first run, the CLI prompts you interactively for:

- Your **HuggingFace token**
- (Optional) email address
- **GPU ID** if you have multiple GPUs
- Whether to run inside **Docker** (recommended)

Your answers are saved to a config file. On subsequent runs of `python3 agora_cli.py`, the saved values are reused; running the command again resumes from the saved configuration.

!!! tip "Using Claude Code? Run `/agora-join`"
    The repo includes a Claude Code skill at `.claude/skills/agora-join` that covers token entry, port mapping, Docker vs native, multi-GPU launch, and startup monitoring. Type `/agora-join` in Claude Code from the repo root to use it instead of the steps below.

---

## Docker vs Native

**Docker** is recommended: it pins the exact Python version and dependencies, isolating the node from the system Python.

If Docker is not available, native install works but requires **Python 3.13 exactly** with **`pip >= 25.3`**: older pip cannot apply the `--build-constraint` used by the install.

```bash
python3 -m pip install --upgrade 'pip>=25.3'
```

!!! warning "RTX 5090 (Blackwell): pre-install CUDA 12.8 PyTorch"
    The PyTorch wheels bundled by the default install do not support Blackwell yet, so 5090-series cards have to install CUDA 12.8 wheels first. Run this before `agora_cli.py`:

    ```bash
    pip install torch==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128
    ```

!!! info "Keep the node running after you log out"
    On any remote SSH machine, run inside [`tmux`](https://github.com/tmux/tmux/wiki) or [`screen`](https://www.gnu.org/software/screen/) so the node does not terminate when your session ends:

    ```bash
    tmux new -s agora
    python3 agora_cli.py
    # Ctrl-b d to detach; `tmux attach -t agora` to reattach
    ```

    Required on RunPod: Docker-in-Docker is impractical there, so use the native install inside tmux. See [Cloud Options → RunPod](setup-guides/cloud.md#runpod).

---

## Expected startup output

Startup runs through four phases. The log signatures below distinguish a healthy startup from one that has stalled in an earlier phase.

### 1. Network check and weight download

```
[NETWORK]  Running internet speed test...
[DOWNLOAD] Downloading model weights...
[DOWNLOAD] Model weights downloaded. Waiting for authorization...
```

If the speed test fails or the download is too slow, your node is dropped from the queue. Check your connection meets the 200 Mbps minimum.

### 2. Authorization queue

```
[AUTH]     Authorization queue: position 2, estimated wait: 1m
[AUTH]     Access granted for your_user
```

During high demand, the queue can be long. If your node times out waiting, restart; it will re-enter the queue.

### 3. Sync (if joining an active run)

```
[SYNC] Synchronising weights with peers. Node won't process batches in this phase.
[SYNC] This phase will last 400 steps (until local epoch <E>).
[SYNC] Synchronising optimizer state. Node is now processing batches, but doesn't contribute to weight averaging yet.
[SYNC] This phase will last 100 steps (until local epoch <E>).
[SYNC] Sync complete. Node is now fully contributing to training.
```

Sync runs in two phases. The first synchronizes your weights with other workers in your stage; the second warms up your local optimizer state. Expect a runtime of several hours end-to-end. Compute points start accruing in the second phase; presence points credit from the moment your peer joins. See [Contributor Join Flow](../agora-system/startup-sequence.md) for the detailed description.

### 4. Training

```
[SERVER]   Training started
[TRAINING] Training step 1
[PROGRESS] Processed 51 batches in the last 60s
[PROGRESS]   Forward pass: 28 batches
[PROGRESS]   Backward pass: 23 batches
```

**Healthy signal:** `[PROGRESS] Processed [N] batches` should update every 60 seconds. As long as that line continues to appear, your node is actively contributing to the swarm.

Detailed logs are written to `logs/server_gpu<ID>.log`.

---

## Verifying your contribution

Once training starts, your participation appears on the [Dashboard](https://dashboard.pluralis.ai/). Look for your node by HuggingFace username.

---

## Stopping the node

- **Native:** press `Ctrl + C` in the terminal running the CLI
- **Docker:** `docker stop <container_name> && docker rm <container_name>`

Rejoining is available at any time. Contribution history is preserved across restarts as long as the same `private.key` file is kept (see [Advanced → private.key](advanced.md#privatekey)).

---

## Troubleshooting

- **Error messages at startup**: check [Setup Guides](setup-guides/cloud.md), which cover auth failures, port tests, CUDA issues, and more
- **Node joined but is not processing batches**: usually a network issue; verify port 49200 is reachable from outside
- **Ask for help**: [Zulip](https://community.pluralis.ai/#narrow/channel/26-A---Contributor-Support)

---

## Advanced
- Multiple GPUs, CLI flags, and manual installation → [Advanced](advanced.md)
