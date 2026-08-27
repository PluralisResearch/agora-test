---
icon: material/rocket-launch
---

# Agora Client

The quick-start has three sections:

1. **[Requirements](requirements.md)**: hardware, OS, network, and the required HuggingFace token
2. **[Running Agora](run.md)**: the full run flow, what logs to expect, how to stop and restart
3. **[Advanced](advanced.md)**: multi-GPU, CLI flags, manual installation, caveats

See [Cloud Options](setup-guides/cloud.md) for running Agora on AWS, GCP, RunPod, Tensordock, or Lambda Labs.

---

## Summary

With the [requirements](requirements.md) met and port **49200** open, start the node with:

```bash
git clone https://github.com/PluralisResearch/agora
cd agora
python3 agora_cli.py
```
The CLI will guide you through the setup process.

!!! info "Check run status before joining"
    Live wait times are on the [Dashboard](https://dashboard.pluralis.ai/) in the **Overview** tab.

Progress is logged live on the [Dashboard](https://dashboard.pluralis.ai/).
