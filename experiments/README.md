# experiments/

Three experiments, one file or package each. Every one of them declares which
runs make up its table, knows how to launch them, and knows how to score them:

    source env/example.sh              # your machine's exports, see README
    python experiments/<experiment>.py --launch [--dry-run]
    python experiments/<experiment>.py --table

| file | what it produces |
|---|---|
| `afseq.py` | the architecture table on AFSeq: 6 architectures x 3 splits, plus persistence |
| `wstsplus.py` | UTAE-AR on WSTS+: 3 feature sets x 4 published folds x 2 history regimes |
| `wstsplus_score.py` | test AP per fold of the published UTAE-AR anchors, for the three feature sets |
| `cross/` | the cross-benchmark table on the AFSeq and WSTS+ fusion in Veg space, plus the native WSTS+ baseline |
| `runner.py` | the shared job runner: lanes, one GPU, resumable, hard CPU ceiling |

Inside `cross/`, `cross.py` is the experiment itself and the rest is what it
needs: `channels.py` maps AFSeq into the WSTS+ Veg channel space, `fusion.py`
defines the fused splits, `native.py` trains the WSTS+-only baseline,
`train_theirs.py` runs their Trainer on the fused data, `eval_test.py` scores
the checkpoints on each benchmark's own test set, and `persist_wsts.py` is the
persistence floor per fold.

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
- **A cap on the processes per GPU**, counted with nvidia-smi before every
  launch (`WSP_MAX_GPU_PROCS`): the day two queues added up to more than the
  cluster tolerated, everything we had was killed.
- **Few lanes** (`WSP_LANES`, default 4) and a **hard CPU ceiling**: every job
  is pinned with `taskset -c $WSP_CPUS` (default `0-15`) and gets the six
  thread variables set to `WSP_THREADS`. Numba alone would start one thread per
  core otherwise. We were twice asked to use fewer cores.
- **Set `AFSEQ_CACHE`** for anything that trains on AFSeq. Without it a run is a
  GDAL benchmark: almost all of the time per sample is spent inside rasterio,
  which opens several `.tif` per day of history, and that holds even with the
  whole dataset in the page cache, so the disk was never the problem.
  `dataset/afseq_cache.py --build` decodes it once, and `--check` proves the
  outputs are `np.array_equal` to the raw loader on random samples of every
  section. With the cache an epoch costs a fraction of what it did and the GPU
  becomes the bottleneck it should be, so two lanes saturate it.
- **Resumable.** A job writes a marker only on exit 0; relaunching skips what is
  already done. These queues run for hours and do get interrupted.

## Scoring is not training

`--table` never retrains. It loads the checkpoints `training/train.py` wrote and
scores them under one protocol stated at the top of each file. `afseq.py` also
re-checks, for every model as it scores it, that appending padded frames does
not change its output. Four of the five architectures once failed that, and
the table rows measured something other than what they claimed.
