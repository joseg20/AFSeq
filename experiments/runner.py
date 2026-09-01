"""The job runner every experiment shares.

Each experiments/expN.py declares WHICH runs make up its table; this module
runs them. Keeping the two apart is what makes an experiment reproducible from
the release alone: the matrix is code that lives next to the paper, not a shell
script in somebody's scratch directory.

WHY IT LOOKS LIKE THIS. The constraints are the cluster's, not PyTorch's:

  one GPU        We are asked to use GPU 0 only, with AT MOST 4 processes on
                 it. The others are left alone even when nvidia-smi shows
                 them idle. The 4 is enforced by counting (MAX_GPU_PROCS).
  few lanes      What runs out first is CPU. We were twice asked to use
                 fewer cores, so LANES is small on purpose and each job
                 gets OMP_NUM_THREADS=2. Without AFSEQ_CACHE the AFSeq loader
                 spends nearly all of its time decoding .tif in rasterio (not
                 disk: iowait is zero and the dataset is in page cache) while
                 the GPU sits mostly idle
                 under 3-4 trainings; with it the GPU saturates at 2 lanes.
                 See dataset/afseq_cache.py.
  resumable      A job writes a marker only when it exits 0, and the runner
                 skips anything already marked. Stopping and relaunching costs
                 at most the jobs that were mid-flight, which matters because
                 these queues run for hours and get interrupted.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LANES = int(os.environ.get("WSP_LANES", 4))
GPU = os.environ.get("WSP_GPU", "0")
THREADS = os.environ.get("WSP_THREADS", "2")

# HARD ceiling on processes in the GPU, counted, not assumed. The rule we were
# given is "at most 4 processes on GPU 0". On 25/08 two queues ran side by
# side, the count reached 5, and every training we had was SIGKILLed within a
# minute. LANES bounds one queue; this bounds all of them together: before a
# lane starts a job it counts our processes on the GPU with nvidia-smi and
# waits while there are MAX_GPU_PROCS or more.
MAX_GPU_PROCS = int(os.environ.get("WSP_MAX_GPU_PROCS", 4))

# HARD ceiling on CPU, not an average. OMP_NUM_THREADS alone is not enough:
# numba (which UMAP pulls in) defaults to one thread per core, 256 here,
# and MKL/OpenBLAS have their own knobs. Pinning every job to the same small
# CPU set with taskset makes the ceiling arithmetic instead of hopeful: nothing
# this runner launches can occupy more than WSP_CPUS cores, whatever a library
# decides at import time. We were asked twice to use fewer cores; a promise
# that depends on every dependency behaving is not a promise.
CPUS = os.environ.get("WSP_CPUS", "0-15")

# Seconds a lane holds the launch lock after starting a job. Counting with
# nvidia-smi is only a ceiling if the count can see the job it just started:
# a training takes 30-60 s to touch the GPU, and two lanes waiting on the same
# GPU used to see the same stale count and start together, 3 processes on a
# GPU capped at 2 (26/08, GPU 1 and GPU 3). Serialising count -> launch -> grace
# closes that window.
LAUNCH_GRACE = int(os.environ.get("WSP_LAUNCH_GRACE", 90))
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")


def our_gpu_procs(gpu: str = GPU) -> int:
    """How many of OUR processes hold memory on `gpu` right now."""
    import getpass
    try:
        uuid = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=uuid",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout.strip()
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout
        mine = set(subprocess.run(
            ["ps", "-u", getpass.getuser(), "-o", "pid="],
            capture_output=True, text=True, timeout=20).stdout.split())
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in apps.splitlines()
               if line.startswith(uuid) and line.split(",")[1].strip() in mine)


def _taskset(cmd: list[str]) -> list[str]:
    """Pin to WSP_CPUS if taskset exists; otherwise run unpinned and say so."""
    if shutil.which("taskset") is None:
        return cmd
    return ["taskset", "-c", CPUS] + cmd


@dataclass
class Job:
    """One training run: a name, a command, and where its log goes."""
    name: str
    cmd: list[str]
    log: Path
    cwd: Path | None = None
    env: dict = field(default_factory=dict)
    # File the job writes only when it finishes. If it is already there, the
    # run happened, even if its marker is missing because the queue was
    # rebuilt or the job predates it. Without this, changing how the queue is
    # launched silently re-runs hours of finished training.
    output: Path | None = None

    def marker(self, done_dir: Path) -> Path:
        return done_dir / self.name.replace("/", "_")

    def already_done(self, done_dir: Path) -> bool:
        if self.marker(done_dir).exists():
            return True
        if self.output is not None and self.output.exists():
            self.marker(done_dir).touch()      # adopt the earlier run
            return True
        return False


def run_queue(jobs: list[Job], done_dir: Path, lanes: int = LANES,
              gpu: str = GPU, threads: str = THREADS, dry_run: bool = False,
              only: list[str] | None = None):
    """Run `jobs` in order, `lanes` at a time, on one GPU. Resumable.

    `only` keeps the jobs whose name contains any of the given substrings.
    It exists for one situation: part of a matrix is already running outside
    this queue (launched by hand, or by an earlier queue that was stopped) and
    relaunching the whole experiment would start it a second time. The
    filter names what to run; everything else is left untouched, not marked.
    """
    done_dir.mkdir(parents=True, exist_ok=True)
    if only:
        jobs = [j for j in jobs if any(o in j.name for o in only)]
        if not jobs:
            raise SystemExit(f"--only {only}: ningun trabajo coincide")
    pending = [j for j in jobs if not j.already_done(done_dir)]
    skipped = len(jobs) - len(pending)

    print(f"{len(jobs)} trabajos, {skipped} ya hechos, {len(pending)} por hacer")
    pin = CPUS if shutil.which("taskset") else "SIN taskset: sin techo duro"
    print(f"{lanes} carriles, GPU {gpu}, {threads} hilos/proceso, CPUs {pin}")
    if dry_run:
        for i, j in enumerate(pending):
            print(f"  {i:>3}  carril {i % lanes}  {j.name}")
            print(f"       {' '.join(j.cmd)}")
        return

    lock = threading.Lock()
    launch = threading.Lock()      # count -> launch -> grace, one lane at a time
    results: dict[str, int] = {}

    def lane(start: int):
        i = start
        while i < len(pending):
            j = pending[i]
            j.log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            for var in _THREAD_VARS:
                env[var] = threads
            env.update(j.env)
            waited = False
            with launch:
                while our_gpu_procs(gpu) >= MAX_GPU_PROCS:
                    if not waited:
                        with lock:
                            print(f"[{datetime.now():%H:%M:%S}] {j.name}: esperando, "
                                  f"ya hay {MAX_GPU_PROCS} procesos nuestros en la "
                                  f"GPU {gpu}", flush=True)
                        waited = True
                    time.sleep(30)
                with lock:
                    print(f"[{datetime.now():%H:%M:%S}] -> {j.name}", flush=True)
                # append, not truncate: relaunching a queue used to overwrite the
                # log of the previous attempt, so a half-finished run could show
                # "Epoch 19/20" from a run that no longer existed
                fh = open(j.log, "a")
                fh.write("\n===== %s %s\n" % (
                    datetime.now().strftime("%F %H:%M:%S"), " ".join(j.cmd)))
                # flush before handing the fd to the child: our write sits in
                # Python's buffer while the subprocess writes straight to the
                # descriptor, so without this the header lands after the run's
                # output, or not at all. That is why relaunching a queue looked
                # like it was truncating logs when it was not.
                fh.flush()
                proc = subprocess.Popen(_taskset(j.cmd), stdout=fh,
                                        stderr=subprocess.STDOUT,
                                        cwd=str(j.cwd) if j.cwd else None, env=env)
                time.sleep(LAUNCH_GRACE)   # until nvidia-smi can count it
            rc = proc.wait()
            fh.close()
            if rc == 0:
                j.marker(done_dir).touch()
            with lock:
                results[j.name] = rc
                print(f"[{datetime.now():%H:%M:%S}] {j.name} rc={rc}", flush=True)
            i += lanes

    threads_ = [threading.Thread(target=lane, args=(k,)) for k in range(lanes)]
    for t in threads_:
        t.start()
    for t in threads_:
        t.join()

    bad = {n: rc for n, rc in results.items() if rc != 0}
    print(f"\nlisto: {len(results) - len(bad)} ok, {len(bad)} con error")
    for n, rc in bad.items():
        print(f"  FALLO rc={rc}  {n}")
    return results
