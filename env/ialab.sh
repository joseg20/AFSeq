# Paths of THIS cluster (IALab / kraken). `source env/ialab.sh` before launching.
# Everything lives in scratch: the home is 50 GB and for code only.
export WSP_DATA=/home/josegui/scratch/afseq/data
export AFSEQ_DATA=$WSP_DATA/dataset_v7
export AFSEQ_CACHE=$WSP_DATA/dataset_v7_cache
export WSTS_HDF5=/home/josegui/scratch/wstsplus/hdf5
export RUNS_DIR=/home/josegui/scratch/afseq/runs
export WSP_GPU=0 WSP_THREADS=2 WSP_MAX_GPU_PROCS=4
# WSP_CPUS: the runner pins with `taskset -c` and ABSOLUTE core ids, so the
# 0-15 default is wrong under Slurm -- the cgroup hands out whichever cores it
# likes and taskset would either fail or pin outside the allocation. Ask the
# cgroup what we actually got.
export WSP_CPUS=$(taskset -cp $$ 2>/dev/null | awk -F': ' '{print $2}')
: "${WSP_CPUS:=0-15}"
export WSP_CPUS
# One GPU per Slurm job, so CUDA_VISIBLE_DEVICES already makes it device 0.
# Lanes: 2 is what saturates a card here once AFSEQ_CACHE is built.
export WSP_LANES=${WSP_LANES:-2}
