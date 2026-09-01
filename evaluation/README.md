# evaluation/: how things are measured

One implementation of the measurement protocol, shared by every experiment.
Reimplementing this per experiment is how you end up with two numbers for the
same model: scoring the whole scene instead of the centre crop moves UTAE-AR on
WSTS+ by more than the gaps the tables resolve, and both are "right" under their own
protocol.

## metrics.py

`pooled_ap_gpu(y, s)` is pooled average precision, computed on the GPU.

Same definition as `sklearn.metrics.average_precision_score`, ties collapsed the
same way, verified equivalent on float16, sparse and saturated inputs, to a
worst-case difference far below any digit we report.

Measured on one fold's worth of data, which is the real size, the two agree to
every digit worth reporting and the GPU path returns in a small fraction of the
time. **It is also far kinder to the machine**: the CPU version held every
thread it could take while the GPU sat idle, sorting that many values is what a
GPU is for, and the peak VRAM is nothing on a modern card.

Nine files of the previous release computed AP with sklearn, including both
training loops, where it runs once per validation pass. They should all call
this instead.

## What else belongs here

The rest of the measurement contract, so no experiment can drift from it:

- the centre crop of the transfer table, versus whole-scene scoring
- the persistence baselines of each benchmark, on the shipped splits
- how folds are averaged, and why the two benchmarks are never pooled: WSTS
  contributes most of the pixels, so micro-pooling and macro-averaging give
  opposite answers about which model is better
- the covariate contract at inference (`observed` / `persist` / `gfs_fcst` /
  `gfs_anl`), which moves the multi-day numbers appreciably and is
  therefore part of the protocol, not an implementation detail
