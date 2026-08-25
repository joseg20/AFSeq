# evaluation/ — how things are measured

One implementation of the measurement protocol, shared by every experiment.
Reimplementing this per experiment is how you end up with two numbers for the
same model: scoring the whole scene instead of the 128-px center crop moves
UTAE-AR on WSTS+ from 0.429 to 0.386, and both are "right" under their own
protocol.

## metrics.py

`pooled_ap_gpu(y, s)` — pooled average precision, computed on the GPU.

Same definition as `sklearn.metrics.average_precision_score`, ties collapsed the
same way, verified equivalent on float16, sparse and saturated inputs (worst
difference 4.4e-08, far below any digit we report).

Measured on one fold's worth of data — 80M pixels, the real size:

| | tiempo | AP |
|---|---|---|
| sklearn, CPU | 21.4 s | 0.242285 |
| `pooled_ap_gpu` | 0.2 s | 0.242285 |

**97x faster, and it frees the cores instead of pinning them.** The CPU version
held 16 threads while the GPU sat at 4%: sorting 80M values is what a GPU is
for. Peak VRAM 3.2 GB, nothing on an 80 GB card.

Nine files of the previous release computed AP with sklearn — including both
training loops, where it runs once per validation pass. They should all call
this instead.

## What else belongs here

The rest of the measurement contract, so no experiment can drift from it:

- the 128-px center crop of Table 3, versus whole-scene scoring
- persistence baselines (AFSeq 0.156 on the v7 splits; 0.252 with all 1,301
  events on the old contour windows; WSTS 0.193)
- how folds are averaged, and why the two benchmarks are never pooled: WSTS
  contributes 91% of the pixels, so micro-pooling and macro-averaging give
  opposite answers about which model is better
- the covariate contract at inference (`observed` / `persist` / `gfs_fcst` /
  `gfs_anl`), which changes multi-day numbers by up to 0.050 AP at t+5 and is
  therefore part of the protocol, not an implementation detail
