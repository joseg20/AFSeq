# Paths of THIS machine (ranokau). `source env/ranokau.sh` before launching.
export WSP_DATA=/workspace1/users/josegui/data
export AFSEQ_DATA=$WSP_DATA/dataset_v7 AFSEQ_CACHE=$WSP_DATA/dataset_v7_cache
export WSTS_HDF5=$WSP_DATA/wstsplus/wsts8/hdf5 RUNS_DIR=/workspace1/users/josegui/runs_v7
# CPUs: user.slice AllowedCPUs=10-255 here (0-9 are the system's), so 0-15 would
# intersect to 6 cores. GPU 1 hangs off NUMA node 0 (CPUs 0-63,128-191); 10-63 is
# 54 threads on 54 distinct physical cores, ~27 per lane. 137 of the allowed
# threads were >85% idle when measured (26/08). 12 workers to actually use them:
# with 4 the GPU sat at 5% and an epoch took 7.3 min.
# 26/08 10:54: capped to 36 threads per queue: GPU-bound, ~10 cores really used.
# GPUs 0-3 hang off NUMA node 0 (CPUs 0-63,128-191). Keep the queue on that node: with CPUs 64-99
# (node 1) the GPU 3 queue sat at ~45% util and 22 min/epoch; on 46-63,128-145 it holds 87-100%.
# Same for GPU 4 (NUMA 1): workers on 64-99 (its own node) still gave 23 min/epoch and a starved GPU;
# on 146-181 (node 0) it is GPU-bound again. The decoded cache lives in node-0 page cache: keep ALL
# loader workers on node 0 whatever the GPU.
#   GPU 1 queue: 10-45   GPU 3 queue (exp3): 46-63,128-145   GPU 4 queue (exp2 ar4/res18): 146-181
export HDF5_USE_FILE_LOCKING=FALSE WSP_THREADS=2 WSP_CPUS=10-45 WSP_WORKERS=12
export WSP_GPU=1 WSP_MAX_GPU_PROCS=2 WSP_LANES=2
