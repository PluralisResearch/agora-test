---
icon: material/cog
---

# Advanced

---

## Multiple GPUs

Run one CLI instance per GPU:

```bash
python3 agora_cli.py --gpu_id 0
python3 agora_cli.py --gpu_id 1
```

Each GPU automatically uses its own port (49200, 49201, ...). Make sure every port is open for inbound TCP.

- **Native.** Run each command in a separate terminal (or tmux window).
- **Docker.** Each instance gets its own container automatically.

!!! info "Maximum 2 peers per account"
    We allow up to **2 active peers per account**. We may relax the cap during the live run.

---

## CLI reference

The CLI prompts interactively, but every value can be passed as a flag:

| Flag | Description |
|------|-------------|
| `--gpu_id <ID>` | GPU to use (default: 0) |
| `--token <TOKEN>` | HuggingFace token |
| `--email <EMAIL>` | Email address (optional) |
| `--host_port <PORT>` | Listening port (default: `49200 + gpu_id`) |
| `--announce_port <PORT>` | External port if different from `--host_port` (e.g. RunPod, Tensordock) |
| `--announce_ip <IP>` | **Advanced.** Public IPv4 address other peers use to reach you. By default the CLI auto-detects it; on some marketplace instances detection can return an address peers cannot reach. Optional: leave unset unless the auto-detected address is wrong, and use the external IP shown in your provider's networking panel. |
| `--use_docker` | Run inside Docker |
| `--log_file <PATH>` | Log file path (default: `logs/server_gpu<ID>.log`) |
| `--identity_path <PATH>` | Identity key path (default: `private_gpu<ID>.key`) |
| `--batch_size_override <N>` | **Advanced.** Force a lower max batch size than the system would auto-pick. Useful when CUDA OOM occurs at the recommended size (common on edge-case GPUs or when other processes share the device). The effective max batch is `min(batch_size_override, recommended_batch_size)`, so the cap can only be lowered, not raised. Start at the recommended size and lower it until OOM stops. |
| `--skip_input` | Non-interactive: all values must come from flags or saved config |
| `--reconfigure` | Re-prompt every setting from scratch |

The `--log_file` and `--identity_path` defaults are appropriate for most setups; override only if there is a specific reason.

---

## Manual installation

### 1. Install dependencies

=== "Docker"

    ```bash
    docker build . -t pluralis_agora --label image_version=1
    ```

=== "Native (no Docker)"

    Prerequisites: Python 3.13, `pip >= 25.3`, NVIDIA drivers with CUDA 12.8, conda.

    ```bash
    # Create and activate a Python 3.13 environment
    conda create -y -n agora python=3.13
    conda activate agora

    # Upgrade pip
    pip install --upgrade "pip>=25.3"

    # Install PyTorch with CUDA 12.8 support
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

    # Install the three required packages from source
    pip install --build-constraint constraints.txt -e ./agora_server
    pip install --build-constraint constraints.txt -e ./agora
    pip install --build-constraint constraints.txt -e ./pithos
    ```

### 2. Read runtime parameters

`run.json` at the repo root contains values required by `run_server.py`:

| `run.json` field | `run_server.py` argument |
|------------------|--------------------------|
| `run_config` | `--config` |
| `auth_server` | `--auth_server` |
| `prom_gateway` | `--prom_gateway` |
| `seeds` (array) | `--initial_peers` |

```bash
RUN_CONFIG=$(jq -r '.run_config' run.json)
AUTH_SERVER=$(jq -r '.auth_server' run.json)
PROM_GATEWAY=$(jq -r '.prom_gateway' run.json)
SEEDS=$(jq -r '.seeds[]' run.json)
```

### 3. Get your public IP

```bash
PUBLIC_IP=$(curl -s https://api.ipify.org)
```

### 4. Launch

Default port convention: `49200 + gpu_id`. Each GPU needs its own port.

=== "Docker"

    ```bash
    docker run -d --name agora_gpu<gpu_id> --ipc=host --network=host \
      --gpus device=<gpu_id> \
      -v $(pwd):/home -w /home \
      pluralis_agora \
      bash -c "CUDA_VISIBLE_DEVICES=0 python3.13 agora/src/agora/run_server.py \
        --gpu_id <gpu_id> \
        --config $RUN_CONFIG \
        --token <hf_token> \
        --auth_server $AUTH_SERVER \
        --prom_gateway $PROM_GATEWAY \
        --host_maddrs /ip4/0.0.0.0/tcp/<port> \
        --announce_maddrs /ip4/$PUBLIC_IP/tcp/<port> \
        --initial_peers $SEEDS \
        --email <email> \
        --log_file logs/server_gpu<gpu_id>.log \
        --identity_path private_gpu<gpu_id>.key"
    ```

=== "Native (no Docker)"

    ```bash
    CUDA_VISIBLE_DEVICES=<gpu_id> python3.13 agora/src/agora/run_server.py \
      --gpu_id <gpu_id> \
      --config "$RUN_CONFIG" \
      --token <hf_token> \
      --auth_server "$AUTH_SERVER" \
      --prom_gateway "$PROM_GATEWAY" \
      --host_maddrs /ip4/0.0.0.0/tcp/<port> \
      --announce_maddrs /ip4/$PUBLIC_IP/tcp/<port> \
      --initial_peers $SEEDS \
      --email <email> \
      --log_file logs/server_gpu<gpu_id>.log \
      --identity_path private_gpu<gpu_id>.key
    ```

---

## Caveats

### private.key

On first run, Agora generates a `private.key` file (or `private_gpu<ID>.key` for multi-GPU setups). This is your node's cryptographic identity.

- **Required** for secure communication within the swarm.
- **Keep it private.** Never share or commit it.
- **Tied to your HuggingFace account.** Rejoining with the same `private.key` under a different HF account fails with `This peer_id is already used by another user`.
- **Losing it** means you join as a new identity. Previous swarm contributions stay on the dashboard, but your `node_id` changes.

### Docker file ownership

Files created inside a Docker container are owned by the container's user, not the host user. To modify or delete them from the host:

**Linux:**

```bash
sudo chown -R $USER <path/to/project>
```

This commonly matters for `logs/` and checkpoint files written during a Docker run.
