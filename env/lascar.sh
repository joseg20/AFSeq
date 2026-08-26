# Paths of THIS machine (lascar). `source env/lascar.sh` before launching.
# The code has no knowledge of these; paths.py only reads the variables.
export WSP_DATA=/workspace1-lascar/users/josegui/data
export AFSEQ_DATA=$WSP_DATA/dataset_v7
export AFSEQ_CACHE=/workspace1-ranokau/users/josegui/data/dataset_v7_cache   # ranokau: lascar is full
export WSTS_HDF5=/workspace1-ranokau/users/josegui/data/wstsplus/wsts8/hdf5
export RUNS_DIR=/workspace1-lascar/users/josegui   # tabla2/ exp1/ exp3/ exp4/ live here
# machine budget we were given: GPU 0 only, at most 4 processes on it, 16 CPUs
export WSP_GPU=0 WSP_CPUS=0-15 WSP_THREADS=2 WSP_MAX_GPU_PROCS=4
# Legacy (this cluster only): the Table 2 trained on ALL 1,301 events lives in
# $RUNS_DIR/tabla2_v4 and its splits in ~/WSP/model. To score it again:
#   AFSEQ_SPLITS_DIR=$HOME/WSP/model python experiments/exp2.py --table --splits v4 --out $RUNS_DIR/tabla2_v4
