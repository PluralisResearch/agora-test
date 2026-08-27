---
icon: material/trophy-outline
---

# Points & Leaderboard

Your leaderboard score combines the raw pflops you have contributed with a presence credit for every hour your peers are active in the swarm, summed across every peer running under your HuggingFace identity. This page covers how that number is collected, what the dashboard shows, and how the per-account cap factors in.

---

## Summary

- Every node accrues **compute points** equal to the raw pflops it has processed.
- Every node also accrues **presence points**: a baseline 1 PFLOP for each hour the peer is active in the swarm, independent of hardware.
- Scores are summed across all your peers: one account, multiple peers = additive, up to a per-account cap.
- Live leaderboard on the [Dashboard](https://dashboard.pluralis.ai/), refreshed every few minutes.

---

## How the score is calculated

Every node periodically reports to the metrics server, and two streams feed your score:

- **Compute**: FLOPs processed since the last report are accumulated into your score. Accrual is continuous, so no work is "lost" if you disconnect mid-batch.
- **Presence**: for every hour the peer is registered and visible in the swarm, a baseline **1 PFLOP** is credited to your score, regardless of how fast the GPU is.

Your leaderboard `score` is the sum of compute and presence pflops across every peer running under your HuggingFace identity.

!!! info "No financial incentive yet"
    Points reflect contribution only. There is no financial reward tied to your score at this stage.

---

## What the leaderboard shows

| Column          | What it means                                                                        |
| --------------- | ------------------------------------------------------------------------------------ |
| **Rank**        | Position on `score`, descending. `1` is the top contributor.                         |
| **Username**    | Your HuggingFace username.                                                           |
| **Active time** | Cumulative **hours** your peer machines have been online since you joined.           |
| **Joined**      | Unix timestamp of when your account first appeared.                                  |
| **Score**       | Compute pflops processed plus 1 PFLOP for each active hour.                          |
| **Status**      | `online` or `offline`.                                                               |

The dashboard refreshes every few minutes, so a fresh contribution appears on the public board shortly after it is recorded.

---

## How to maximize your score

- **Faster hardware.** A GPU with more FLOPs per second gives you more compute pflops per minute.
- **Long uptime.** Compute and presence both accrue continuously. Every hour your peer stays active credits 1 baseline PFLOP, so 24/7 operation on one card outscores 4 hr/day on the same card on both fronts.
- **Stable network.** Disconnects do not erase accrued points; they pause accrual until the node is back online.
- **Multiple peers per account.** Scores from peers running under the same identity sum. We allow up to **2 active peers per account**. We may relax the cap during the live run.
