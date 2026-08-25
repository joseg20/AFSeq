# experiments/

One file per experiment. Each declares WHICH runs make up its table, knows how
to launch them, and knows how to score them:

    source env/lascar.sh               # or your machine's exports, see README
    python experiments/expN.py --launch [--dry-run]
    python experiments/expN.py --table

| file | what it produces |
|---|---|
| `exp1.py` | UTAE-AR on WSTS+: 3 feature sets x 4 published folds x 2 history regimes |
| `exp2.py` | the architecture table on AFSeq: 6 architectures x 3 splits, plus persistence |
| `exp3.py` | the teacher-forcing schedule ablation for UTAE-AR on AFSeq |
| `exp4/` | the cross-benchmark table on the AFSeq+WSTS+ fusion (Veg space), plus the native WSTS+ baseline |
| `exp5/` | the two-panel UMAP figure, from the encoder `exp4/native.py` trains |
| `runner.py` | the shared job runner: lanes, one GPU, resumable, hard CPU ceiling |

`--only SUBSTR...` restricts `--launch` to the jobs whose name contains one of
the substrings. It is for the one situation where part of a matrix is already
running outside the queue and relaunching the experiment would start it twice.

## Why the matrix is code

It used to live in a shell script in a scratch directory, which meant the
release could train a model but could not reproduce a table: nothing in it said
which runs a table was made of. Everything needed to go from a clean checkout
to a number is now here.

## What the runner assumes

- **GPU 0 only.** The cluster asks for it; the other cards are left alone even
  when `nvidia-smi` shows them idle. Override with `WSP_GPU`.
- **At most 4 processes on GPU 0**, counted with nvidia-smi before every launch (`WSP_MAX_GPU_PROCS`): the day two queues added up to 5, everything we had was killed.
- **Few lanes** (`WSP_LANES`, default 4) and a **hard CPU ceiling**: every job
  is pinned with `taskset -c $WSP_CPUS` (default `0-15`) and gets the six
  thread variables set to `WSP_THREADS` (default 2). Numba alone would start
  256 threads otherwise. We were twice asked to use fewer cores.
- **Set `AFSEQ_CACHE`** for anything that trains on AFSeq. Without it a run is a
  GDAL benchmark: 186 ms per sample, 89% inside rasterio, 57 `.tif` opens per
  sample, ~4 min per epoch with 4 workers and the GPU at 0-40% -- and that is
  with the whole dataset in the page cache (iowait 0; the disk was never the
  problem). `dataset/afseq_cache.py --build` decodes it once (4.6 GB, one
  minute); `--check` proves the outputs are `np.array_equal` to the raw loader
  on random samples of every section. With the cache a sample costs 5-8 ms,
  UTAE-Res18 goes from 3.5 min to ~30 s per epoch and the GPU becomes the
  bottleneck it should be, so two lanes saturate it.
- **Resumable.** A job writes a marker only on exit 0; relaunching skips what is
  already done. These queues run for hours and do get interrupted.

## Scoring is not training

`--table` never retrains. It loads the checkpoints `training/train.py` wrote and
scores them under one protocol stated at the top of each file. `exp2.py` also
re-checks, for every model as it scores it, that appending padded frames does
not change its output -- four of the five architectures once failed that, and
the table rows measured something other than what they claimed.
