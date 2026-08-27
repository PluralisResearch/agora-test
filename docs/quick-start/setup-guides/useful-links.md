---
icon: material/link-variant
---

# Useful Links

External references for the toolchain Agora relies on. The main guides link into these where relevant; this page collects them for browsing.

---

## Authentication & access

The credentials and SSH setup required before deployment or code checkout.

- [Creating a Hugging Face access token](https://huggingface.co/docs/hub/security-tokens): required for Agora identity
- [Generating an SSH key pair (macOS / Linux)](https://www.digitalocean.com/community/tutorials/how-to-create-ssh-keys-with-openssh-on-macos-or-linux)
- [Connecting to a server over SSH](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server)
- [Adding an SSH key to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [Cloning a GitHub repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
- [Fixing "UNPROTECTED PRIVATE KEY FILE"](https://www.cyberciti.biz/faq/warning-unprotected-private-key-file-ssh-linux-unix-error/): common SSH permissions error

---

## Linux fundamentals

Package management, permissions, and inspecting running processes.

- [`apt` package manager (Debian / Ubuntu)](https://help.ubuntu.com/community/AptGet/Howto)
- [`chmod`, `chown`, and UNIX file permissions](https://www.digitalocean.com/community/tutorials/how-to-set-permissions-linux)
- [Using `lsof` to see open ports and GPU processes](https://linux.die.net/man/8/lsof)

---

## GPU & NVIDIA drivers

- [Installing NVIDIA drivers (Ubuntu)](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
- [Verifying GPU availability with `nvidia-smi`](https://www.gpu-mart.com/blog/monitor-gpu-utilization-with-nvidia-smi)

---

## Environments: Docker & Conda

Agora can run in either; pick one. On RunPod the workspace is already containerised → use conda. On a clean VM either works.

- [Docker Engine install guide (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/)
- [Docker post-install steps](https://docs.docker.com/engine/install/linux-postinstall/): add your user to the `docker` group
- [Miniconda install (Linux)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html): envs, activation, package install

---

## Networking & firewall

For cloud-provider port-opening, see the per-provider sections in [Cloud Options](cloud.md). For laptop / home / WSL2 setups, see [Requirements → Network](../requirements.md#network). The generic Linux references below are useful for VM setups.

- [Opening a TCP port on Linux](https://www.digitalocean.com/community/tutorials/opening-a-port-on-linux)
- [Linux firewall basics: iptables / firewalld](https://www.digitalocean.com/community/tutorials/iptables-essentials-common-firewall-rules-and-commands)
