---
name: agora-join
description: Join the Pluralis Agora distributed training run. Use when the user wants to connect their GPU to the public run, configure ports, and start the training server.
disable-model-invocation: false
argument-hint: "[--reconfigure]"
allowed-tools: Bash, Read, AskUserQuestion
---

# agora-join

Join a GPU to the Pluralis Agora distributed training run. This skill wraps `agora_cli.py` — the existing CLI handles installation, container launch, and config persistence; the skill collects the inputs conversationally, decides the announce-port question, launches in the background, monitors startup logs, diagnoses common failures, and hands off monitoring/stop commands.

The skill is written so any AI coding agent can follow it. Where Claude Code has a specific tool that makes a step cleaner, the prose calls it out — other agents should substitute their own equivalents (a regular text prompt for `AskUserQuestion`, `nohup` for backgrounded `Bash`, etc.).

If the user passes `--reconfigure`, force re-prompting of every saved value and pass `--reconfigure` to `agora_cli.py`.

---

## 1. Goal

Join the user's GPU to the public run by collecting credentials and runtime parameters, launching `agora_cli.py start --skip_input ...` in the background (Docker container or tmux-backed native process), and verifying the server reaches `[SERVER]   Training started` before handing off live-monitoring and stop commands.

## 2. Pre-flight

Run these checks at the start. Stop with a clear message on any hard failure.

- `ls agora_cli.py run.json` from cwd. If either is missing, the user is not in the cloned agora repo. Tell them to `cd` into it, or — if they confirm — offer to `git clone https://github.com/PluralisResearch/agora` and `cd agora`. Do not clone without confirmation.
- `cat run.json` and parse. If `run_config`, `auth_server`, `prom_gateway`, or `seeds` is the literal string `"none"` or an empty array, stop with: "This repo was cloned before a run was published. Pull the latest with `git pull` and re-run, or wait for the next run announcement."
- `nvidia-smi -L`. If it fails (no driver, no GPU), stop with: "NVIDIA driver / GPU not detected. agora needs an NVIDIA GPU with the NVIDIA driver installed."
- `docker info >/dev/null 2>&1` — record whether Docker is available.
- `command -v tmux >/dev/null` — record whether tmux is available (only needed for native mode).
- `cat ~/.agora/user_config.json 2>/dev/null` — load saved config if it exists. Use saved values as defaults in phase 4. Never echo the saved token back to the user.

## 3. Multi-GPU mode

Run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits` and parse rows. Filter to entries with VRAM ≥ 24576 (24 GB). The minimum requirement is 24 GB VRAM; smaller GPUs will be rejected by the server.

- If zero eligible GPUs → stop with "No GPU with ≥24 GB VRAM detected; agora requires at least 24 GB."
- If exactly one eligible GPU → use it silently and remember its index.
- If multiple eligible GPUs → ask the user (Claude Code: `AskUserQuestion`):
  - "Launch on a single GPU?" with that GPU's index/name as the option.
  - "Launch on all N eligible GPUs?" — the skill will run phases 7–10 sequentially for each GPU.
  If "single", ask which index in a follow-up.

## 4. Collect missing user info

For each value not in saved config (or all of them if `--reconfigure`):

- **HuggingFace token** — free-form text prompt in the conversation. Say: "Paste your HuggingFace token. You can create one at https://huggingface.co/settings/tokens with no scopes selected — it's used only for authorization." **Never put a secret in a fixed-choice menu, and never echo the token back in confirmations or summaries.**
- **Email** (optional) — free-form, accept blank input.
- **Docker vs native** — fixed-choice (Claude Code: `AskUserQuestion`). Default to Docker if `docker info` worked. If user picks native, verify both:
  - `python3 --version` reports exactly `3.11.x` — native mode is pinned to Python 3.11; other 3.x versions will not work.
  - `python3 -m pip --version` reports `25.3` or newer — older pip will fail to resolve the constrained installs. If older, run `python3 -m pip install --upgrade 'pip>=25.3'` and re-check.

  If Python 3.11 is missing, help install it for the user or fall back to Docker. If neither Docker nor a working Python 3.11 + pip ≥ 25.3 is available, stop.
- **Native + tmux missing** — only ask if user picked native AND tmux isn't available. Fixed-choice: "tmux is needed to keep the server running after you disconnect from this terminal. Install it now (`sudo apt-get install -y tmux`), or switch to Docker?" Run the install only on confirmation; otherwise switch to Docker (if available) or stop.

Save the collected values back via `agora_cli.py` itself — the CLI persists `~/.agora/user_config.json` on each run. The skill does not write that file directly.

## 5. Port decision

This is the most common point of user-facing failure. Ask one fixed-choice question (Claude Code: `AskUserQuestion`):

> "Is this machine a rented cloud GPU (RunPod / Vast.ai / Tensordock / Salad), inside a Docker container without `--network=host`, or behind NAT with manual port forwarding?"

- **No** → both ports are equal: `host_port = announce_port = 49200 + gpu_id`. Tell the user: "I'll use port `49200 + gpu_id` for both listening and announcing. Make sure that port is open in your firewall / cloud security group for inbound TCP."
- **Yes** → free-form prompt: "What is the externally-reachable port your provider mapped to container port `49200 + gpu_id`? Check your provider's port-mapping panel — RunPod calls it `RandomPort`, Vast.ai shows it under your instance's networking section, etc." Set `host_port = 49200 + gpu_id` and `announce_port = <user value>`.

If `[ -e /.dockerenv ]` is true AND the user said "no NAT", warn: "You appear to be inside a Docker container. If it wasn't started with `--network=host`, you'll need to set the announce port to whatever external port maps to `49200 + gpu_id`. Continue anyway?" — let the user override.

## 6. Pre-launch sanity (warn-only)

Before launching, surface any state that might cause confusion. None of these block launch.

- **Docker mode**: `docker ps -a --filter name=agora_gpu${GPU_ID} --format '{{.Names}}'` — if a stale container exists, tell the user: "An existing container `agora_gpu${GPU_ID}` will be stopped and removed (this is intentional)." (See `agora_cli.py:756-769` — the CLI handles cleanup automatically even with `--skip_input`.)
- **Native mode**: `tmux has-session -t agora_gpu${GPU_ID} 2>/dev/null` — if a stale session exists, ask (Claude Code: `AskUserQuestion`): "Kill existing tmux session `agora_gpu${GPU_ID}` and relaunch?" On confirm: `tmux kill-session -t agora_gpu${GPU_ID}`. (The CLI does not auto-clean tmux sessions like it does Docker containers.)
- **Both modes**: `ss -ltn 2>/dev/null | grep -E ":${HOST_PORT}\b"` — if something is listening on the host port, warn but do not block.

## 7. Launch (background, mode-dependent)

Build the common arg list once. Pass values via shell-quoted argv; HF tokens contain no shell metacharacters in practice but quote anyway.

```bash
ARGS=(
  start --skip_input
  --token "$TOKEN"
  --gpu_id "$GPU_ID"
  --host_port "$HOST_PORT"
  --announce_port "$ANNOUNCE_PORT"
)
[ -n "$EMAIL" ]      && ARGS+=(--email "$EMAIL")
[ -n "$ANNOUNCE_IP" ] && ARGS+=(--announce_ip "$ANNOUNCE_IP")
[ "$USE_DOCKER" = 1 ] && ARGS+=(--use_docker)
[ "$RECONFIGURE" = 1 ] && ARGS+=(--reconfigure)
```

`ANNOUNCE_IP` is empty in the normal flow. It is set only during port-test failure recovery (phase 8), when public IP auto-detection returned an address peers cannot reach. The CLI does not persist it, so every relaunch after that recovery must pass it again.

### Docker path

The CLI invokes `docker run -d` (`agora_cli.py:773-792`) which detaches the container; the CLI itself returns within seconds. Container stdout is auto-redirected to `agora_output_gpu<ID>.log` (`agora_cli.py:771`).

```bash
python3 agora_cli.py "${ARGS[@]}" > "/tmp/agora_cli_gpu${GPU_ID}.out" 2>&1
```

Claude Code: run with `Bash` and `run_in_background: true`. Other agents: `nohup ... &` is fine — the CLI exits in seconds either way.

### Native path (tmux required)

**Install `agora_server` and `agora` first.** The CLI does not install them in native mode (Docker mode bakes them into the image; native does not). From the repo root:

```bash
pip install --build-constraint constraints.txt -e ./agora_server
pip install --build-constraint constraints.txt -e ./agora
```

Both installs are required and must use `--build-constraint constraints.txt` — without it, transitive build deps drift and the server fails to start. If either install fails, surface the pip error and stop — do not proceed to launch.

`_start_native` (`agora_cli.py:806-902`) blocks waiting on the server subprocess and streams its stdout to whatever terminal it inherits. Backgrounding the CLI directly from the agent's shell is unsafe: when the agent's shell or the user's SSH session terminates, SIGHUP propagates and the server dies. Solution: run the CLI inside a **detached tmux session**.

```bash
TMUX= tmux new-session -d -s "agora_gpu${GPU_ID}" -c "$(pwd)" \
  "python3 agora_cli.py ${ARGS[*]@Q}"
```

- `TMUX=` clears the env var so this works whether or not the agent is itself inside tmux.
- `-d` keeps the session detached.
- `-c "$(pwd)"` runs the CLI from the repo root (where `logs/server_gpu<ID>.log` will be written).

If `TMUX= tmux new-session ...` fails (rare, but possible on hosts with broken tmux configs), fall back to:
```bash
nohup setsid python3 agora_cli.py "${ARGS[@]}" \
  > "/tmp/agora_cli_gpu${GPU_ID}.out" 2>&1 < /dev/null &
echo $! > "/tmp/agora_cli_gpu${GPU_ID}.pid"
```
Tell the user the fallback was used and that they'll need to `kill -INT $(cat /tmp/agora_cli_gpu${GPU_ID}.pid)` to stop.

## 8. Monitor — Tier 1 (fail-fast, ≤90 s)

Poll every ~15 s (six iterations max). Read the tail of all relevant sources combined.

**Docker mode** sources:
- `tail -n 200 logs/server_gpu${GPU_ID}.log 2>/dev/null` — structured server log
- `tail -n 100 agora_output_gpu${GPU_ID}.log 2>/dev/null` — container stdout/stderr
- `tail -n 100 /tmp/agora_cli_gpu${GPU_ID}.out 2>/dev/null` — CLI's own stdout (catches `docker pull` / install errors before the container starts)

**Native mode** sources:
- `tail -n 200 logs/server_gpu${GPU_ID}.log 2>/dev/null`
- `tmux capture-pane -t agora_gpu${GPU_ID} -p -S -300 2>/dev/null` — CLI stdout/stderr in the tmux pane (last ~300 lines; catches Python tracebacks, banner, install failures, unstructured server output)
- `tmux has-session -t agora_gpu${GPU_ID} 2>/dev/null` — if the session has disappeared, the CLI exited. Surface the last `capture-pane` output as the failure message.

Grep the combined output (case-sensitive) for the patterns below. Act on the first match.

**Mirrors README.md troubleshooting section — keep both in sync when error messages change.**

| Pattern (regex) | Severity | Action |
|---|---|---|
| `Port test failed` | STOP | Most common failure. Walk the user through firewall / port-forwarding for their cloud provider. Offer to relaunch with a corrected announce port (re-enter phase 5). If the user confirms the port mapping and firewall are correct, the auto-detected public IP is likely wrong (happens on some marketplace hosts): ask for the external IP shown in the provider's networking panel, set `ANNOUNCE_IP` to it, and relaunch so `--announce_ip` is passed (phase 7). |
| `Invalid HuggingFace token` | STOP | Token bad or expired. Re-prompt for token; relaunch with `--reconfigure`. |
| `peer_id is already used by another user` OR `already in the authorization queue. Please remove private key` | STOP | Identity collision. Offer to `rm <repo-root>/private_gpu${GPU_ID}.key` (per `agora_cli.py:709`) and relaunch. |
| `CUDA is not available` | STOP | No CUDA detected inside the runtime. Tell user to verify `nvidia-smi` works inside the runtime they chose (host for native, container for Docker — `docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi`). |
| `GPU is not eligible \(GPU VRAM:` | STOP | This GPU has <24 GB VRAM; cannot proceed on this device. Offer a different GPU index if one is available. |
| `Wrong pytorch version` | STOP | Native-only. Recommend re-running with `--use_docker` (one-line fix). The fallback is `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128`. |
| `Verification failed` | STOP | Local modifications to the source detected. Recommend `git status` and `git stash`, then relaunch. |
| `Download is too slow` | STOP | Bandwidth fail; need ≥200 Mbps. Suggest a different network. |
| `Server exited with -9` OR `RAM usage exceeded threshold` | STOP | OOM. Free RAM; if Docker, raise `--memory`. The machine should have ≥80 GB RAM per GPU. |
| `Too many failed all-reduce attempts` | INFO + STOP | Connection unstable. Suggest retrying on a more stable network. |
| `Authorization timeout after` | INFO + STOP | High load. Re-running the skill is the fix; do **not** delete the private key. |
| `\[DOWNLOAD\] Model weights downloaded` | PROMOTE | Tier 1 cleared — promote to Tier 2 immediately. |
| `\[SERVER\]   Training started` | SUCCESS | Skip to phase 10. |

When you act on a STOP pattern, also surface the relevant tail (~20 lines) so the user sees the actual log context, not just your interpretation.

## 9. Monitor — Tier 2 (auth wait, up to 15 min wall-clock from launch)

Poll every ~30 s. Same combined-tail logic as Tier 1, but only the failure patterns matter and the success pattern is `[SERVER]   Training started` or `[PROGRESS] Processed`.

When the line `[AUTH]     Authorization queue: position <N>, estimated wait: <Xm>` appears, track the last-seen `<N>`. Surface it to the user **only when N changes** (don't spam every poll). Example surface: "You're at position 3 in the auth queue, estimated wait 2m."

If 15 min elapses without success and no STOP pattern hit, **don't error**. Print the hand-off (phase 10) with a note: "You're still in the auth queue (position N as of last poll). Re-run this skill or run `tail -f logs/server_gpu${GPU_ID}.log` to keep watching — auth wait can exceed 15 min during high demand."

## 10. Hand-off

Print mode-specific commands. Keep the lists short — the user reads these once.

### Docker

- Live log: `tail -f logs/server_gpu${GPU_ID}.log` (structured) or `tail -f agora_output_gpu${GPU_ID}.log` (container stdout).
- Status: `docker ps --filter name=agora_gpu${GPU_ID}`.
- Stop: `docker stop agora_gpu${GPU_ID} && docker rm agora_gpu${GPU_ID}`.

### Native

- Live log: `tail -f logs/server_gpu${GPU_ID}.log`. To watch the CLI's live terminal output, attach to the tmux session: `tmux attach -t agora_gpu${GPU_ID}` (detach with `Ctrl-b d` — does **not** stop the server).
- Status: `tmux has-session -t agora_gpu${GPU_ID} && echo running || echo stopped`. Plus `pgrep -af 'run_server.py.*--gpu_id ${GPU_ID}'` to confirm the server subprocess is alive.
- Stop (graceful): `tmux send-keys -t agora_gpu${GPU_ID} C-c` — sends SIGINT to the CLI which propagates to the server's process group (`agora_cli.py:823-853` does the SIGTERM→SIGKILL escalation). Then `tmux kill-session -t agora_gpu${GPU_ID}` to clean up the empty session.
- Force stop (only if graceful hangs): `tmux kill-session -t agora_gpu${GPU_ID}`.

### Both

- Dashboard: https://agora-testing.pluralis.ai/ — your contribution shows up here.
- "Re-run this skill any time to rejoin or reconfigure (pass `--reconfigure` to re-prompt all values)."
- The `private_gpu${GPU_ID}.key` file at the repo root is your node's identity. Keep it; deleting it generates a new identity tied to a fresh node entry.

## 11. Multi-GPU loop

Only if the user picked "all" in phase 3. After the first GPU's hand-off, repeat phases 7–10 for the next GPU index. The same saved config (token, email, docker, NAT-yes-or-no) applies; only `gpu_id`, `host_port`, `announce_port` change. An `--announce_ip` override, if one was needed, is machine-wide: reuse the same `ANNOUNCE_IP` for every GPU.

If the user picked "all" in a NAT/cloud-port-mapping setup, ask in phase 5 for the announce port for **each** GPU separately — cloud providers map each container port independently.

## 12. Reconfigure mid-skill

If the user says "change my token / port / email" mid-conversation, jump back to phase 4 (or phase 5 for ports), re-collect just the changed values, and relaunch passing `--reconfigure`. Do not edit `~/.agora/user_config.json` directly — the CLI owns that file (`agora_cli.py:451-452`) and surgical edits will drift.
