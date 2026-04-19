# Single-beat vs 2-second window — head-to-head

Both autoencoders use the identical architecture (conv encoder + dense
bottleneck + transposed-conv decoder, latent dim 32). Only the input
size differs.

| Setting | Window | Equivalent | Test ROC-AUC | F1 | Sens | Spec |
|---|---|---|---|---|---|---|
| Published (2 s) | 720 sa | ~2 s, ~3 beats | 0.972 | 0.851 | 0.902 | 0.955 |
| **This run (single-beat)** | 256 sa | ~0.71 s, ~1 beat | **0.916** | **0.704** | **0.817** | **0.895** |

Trained for 30 epochs on the same MIT-BIH train/val split
(63391 normals only) and evaluated on the same test split
(16419 beats, of which 2841 abnormal).

## Reading the gap

A +5.6-point ROC-AUC drop and a
+14.7-point F1 drop on the same dataset, with only the
window length changed. The 2-second window's advantage comes from
seeing the **rhythm context** (RR-interval) and **neighbouring beat
morphology** that single-beat windows by definition cannot use.
