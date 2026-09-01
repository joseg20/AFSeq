#!/usr/bin/env python3
"""Experiment 3 -- how fast should teacher forcing decay? (UTAE-AR, AFSeq)

    python experiments/exp3.py --launch
    python experiments/exp3.py --table             # best val AP per schedule
    python experiments/exp3.py --horizons [--json] # TEST AP at t+1..t+5, per schedule

UTAE-AR rolls five days forward, feeding its own prediction back in. During
training it can instead be fed the ground truth -- teacher forcing -- which
makes the loss easier but trains the model in a regime it will never see at
test time. The schedule that decays it is therefore a real hyper-parameter of
the method, not a detail, and this is the ablation of it.

THE SCHEDULES, all over 20 epochs:

  none        tf = 0 from the start. Free running throughout: the model only
              ever sees its own predictions.
  linear      tf = max(0.5, 1 - 0.5 * ep / (E-1)). The paper's default. Note it
              floors at 0.5 -- it never stops teacher forcing entirely.
  ss(k)       tf = k / (k + exp(ep / k)), inverse-sigmoid decay (Bengio et al.
              2015). k sets how long the model is held on ground truth:

                  k     epoch 0   epoch 10   epoch 19
                   1      0.50      0.000      0.000
                   2      0.67      0.013      0.000
                   3      0.75      0.097      0.005
                   5      0.83      0.405      0.084
                  10      0.91      0.785      0.599
                  20      0.95      0.881      0.885

              So k=1 is almost free running, k=20 almost pure teacher forcing,
              and the interesting region is 3-10 where the crossover happens
              inside the run.

WHY AP PER HORIZON, not just t+1. Teacher forcing cannot affect the first step:
day t+1 is predicted before fut_covars or any fed-back mask is touched, so
every schedule gives an identical t+1 by construction. If a table of this
ablation shows t+1 moving, something else changed. The effect lives at t+2..t+5,
where the errors compound, which is exactly what the schedule is meant to
control.

SEEDS. One split (s23) by default: this is an ablation over 8 configurations,
and 3 splits would be 24 runs for a curve whose shape, not whose third decimal,
is the point. Use --seeds for the final version if a number goes in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from runner import Job, run_queue                              # noqa: E402
from paths import RUNS, SPLITS_DIR                             # noqa: E402

KS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
MODES = ("none", "linear")          # the two non-ss baselines
SPLIT_DIR = SPLITS_DIR                 # dataset/splits, shipped with the code
EPOCHS = 20
LOSS = "bce_dice"
LR = 1e-3
SEED_DEFAULT = 23


def tf_at(mode: str, k: float, ep: int, epochs: int = EPOCHS) -> float:
    """The schedule, copied from training/train.py so --table can print it."""
    if mode == "none":
        return 0.0
    if mode == "ss":
        return float(k / (k + math.exp(ep / k)))
    return max(0.5, 1.0 - 0.5 * ep / max(1, epochs - 1))


def tag_of(mode: str, k: float, seed: int) -> str:
    """The tag training/train.py writes for this configuration."""
    lr_tag = f"{LR:g}".replace("-", "").replace(".", "p")
    tag = f"utae_ar_{LOSS}_lr{lr_tag}"
    if mode != "linear":
        tag += f"_tf{mode}" + (f"k{k:g}" if mode == "ss" else "")
    if seed != SEED_DEFAULT:
        tag += f"_s{seed}"
    return tag


def configs(ks=KS, modes=MODES):
    for mode in modes:
        yield mode, 0.0
    for k in ks:
        yield "ss", k


def matrix(out: Path, seeds, ks=KS, modes=MODES) -> list[Job]:
    jobs = []
    for seed in seeds:
        for mode, k in configs(ks, modes):
            tag = tag_of(mode, k, seed)
            cmd = [sys.executable, str(ROOT / "training" / "train.py"),
                   "--model", "utae_ar", "--loss", LOSS, "--lr", str(LR),
                   "--epochs", str(EPOCHS), "--seed", str(seed),
                   "--tf-mode", mode, "--workers", "4", "--out", str(out)]
            if mode == "ss":
                cmd += ["--k", f"{k:g}"]
            jobs.append(Job(name=tag, cmd=cmd,
                            log=out / "logs" / f"{tag}.log", cwd=ROOT,
                            output=out / f"{tag}.json",
                            env={"AFSEQ_SPLITS":
                                 str(SPLIT_DIR / f"splits_s{seed}.txt")}))
    return jobs


def table(out: Path, seeds, ks=KS, modes=MODES):
    print(f"{'schedule':<12}{'tf ep0':>8}{'tf ep19':>9}{'val AP':>10}{'seeds':>7}")
    for mode, k in configs(ks, modes):
        vals = []
        for seed in seeds:
            js = out / f"{tag_of(mode, k, seed)}.json"
            if js.exists():
                vals.append(json.loads(js.read_text())["best_val_ap"])
        if not vals:
            continue
        name = mode if mode != "ss" else f"ss k={k:g}"
        cell = (f"{statistics.mean(vals):.4f}" if len(vals) == 1 else
                f"{statistics.mean(vals):.4f}+-{statistics.stdev(vals):.3f}")
        print(f"{name:<12}{tf_at(mode, k, 0):>8.2f}{tf_at(mode, k, EPOCHS-1):>9.2f}"
              f"{cell:>10}{len(vals):>7}")
    print("\nRecordatorio: t+1 es identico para todos los horarios por")
    print("construccion; el efecto esta en t+2..t+5.")


# ------------------------------------------------------- AP per horizon (test)
HORIZON = 5


def _test_loader(splits_file, workers=4, batch_size=4):
    """exp2's test loader, but returning the full 5-day target (horizon=HORIZON)."""
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    from paths import AFSEQ
    from dataset import afseq_loader
    from dataset.afseq_cache import install_if_configured
    from dataset.afseq_loader import (PATCH, Seq2SeqActivePadded, collate_padded,
                                      dl_ids, load_ids)
    install_if_configured(afseq_loader)    # only if AFSEQ_CACHE is set
    ids = dl_ids(load_ids("Test", splits_file), AFSEQ)
    ds = Seq2SeqActivePadded(ids, str(AFSEQ), PATCH, 16, HORIZON)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      collate_fn=lambda b: collate_padded(b, 0.0))


def score_horizons(model, loader, dev):
    """Pooled AP at every t+k of a free-running 5-day rollout on test.

    Same protocol as exp2.score (pooled, valid samples only), extended along
    the horizon: the k-th step is scored on the samples whose k-th target day
    exists (V[:, k]), so later horizons pool fewer samples -- the count is
    returned with the AP. Free running: y_true=None, teacher forcing 0.
    """
    import torch
    from evaluation.metrics import pooled_ap_gpu
    ys = [[] for _ in range(HORIZON)]
    ps = [[] for _ in range(HORIZON)]
    # threshold-free AP can reward a diffuse, well-ordered forecast that is not a
    # better mask. These sums answer whether the gain is hedging: predicted
    # probability mass vs burned pixels, area at 0.5 vs burned area, F1/IoU at
    # 0.5, Brier.
    acc = [dict(sp=0.0, sy=0.0, pp=0.0, tp=0.0, brier=0.0, n=0) for _ in range(HORIZON)]
    with torch.no_grad():
        for X, FC, Y, V, _ in loader:
            X, FC = X.to(dev, non_blocking=True), FC.to(dev, non_blocking=True)
            Y, V = Y.to(dev, non_blocking=True), V.to(dev, non_blocking=True)
            out = model(X, FC, y_true=None, teacher_forcing_ratio=0.0)   # (B,H,1,h,w)
            if isinstance(out, tuple):
                out = out[0]
            for k in range(HORIZON):
                keep = V[:, k] > 0.5
                if not keep.any():
                    continue
                y = (Y[keep, k, 0] > 0.5).flatten()
                p = torch.sigmoid(out[keep, k, 0]).flatten()
                ys[k].append(y)
                ps[k].append(p)
                a = acc[k]
                yf = y.float()
                a["sp"] += float(p.sum()); a["sy"] += float(yf.sum())
                hard = p > 0.5
                a["pp"] += float(hard.sum()); a["tp"] += float((hard & y).sum())
                a["brier"] += float(((p - yf) ** 2).sum()); a["n"] += int(y.numel())
    res = []
    for k in range(HORIZON):
        if not ys[k]:
            res.append(dict(ap=float("nan"), pixels=0))
            continue
        y, p = torch.cat(ys[k]), torch.cat(ps[k])
        a = acc[k]
        prec = a["tp"] / max(a["pp"], 1.0)
        rec = a["tp"] / max(a["sy"], 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        iou = a["tp"] / max(a["pp"] + a["sy"] - a["tp"], 1.0)
        res.append(dict(ap=pooled_ap_gpu(y, p, device=dev), pixels=int(y.numel()),
                        mass_ratio=a["sp"] / max(a["sy"], 1.0),     # sum p / burned px
                        area_ratio=a["pp"] / max(a["sy"], 1.0),     # px with p>0.5 / burned px
                        precision=prec, recall=rec, f1=f1, iou=iou,
                        brier=a["brier"] / max(a["n"], 1)))
    return res


def horizons(out: Path, seeds, ks=KS, modes=MODES, workers=4, batch_size=4):
    """TEST AP at t+1..t+5 for every trained schedule, one row per (schedule, seed)."""
    import torch
    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "training"))
    from train import build_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = {s: _test_loader(str(SPLIT_DIR / f"splits_s{s}.txt"), workers,
                               batch_size) for s in seeds}
    res = {}
    for mode, k in configs(ks, modes):
        name = mode if mode != "ss" else f"ss k={k:g}"
        for seed in seeds:
            ck = out / f"best_{tag_of(mode, k, seed)}.pth"
            if not ck.exists():
                print(f"  {name:11s} s{seed}: sin checkpoint todavia")
                continue
            model = build_model("utae_ar", HORIZON)
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model = model.to(dev).eval()
            r = score_horizons(model, loaders[seed], dev)
            res.setdefault(name, {})[seed] = dict(
                tf0=tf_at(mode, k, 0), tf19=tf_at(mode, k, EPOCHS - 1), horizons=r)
            print(f"  {name:11s} s{seed}: " + "  ".join(
                f"t+{i + 1} {h['ap']:.4f} ({h['pixels'] / 1e6:.1f}M px)" for i, h in enumerate(r)))
            del model
            torch.cuda.empty_cache()

    print(f"\n{'schedule':<12}{'tf ep0':>8}{'tf ep19':>9}" +
          "".join(f"{'t+' + str(i + 1):>9}" for i in range(HORIZON)) + f"{'seeds':>7}")
    for name, per_seed in res.items():
        any_ = next(iter(per_seed.values()))
        cells = []
        for i in range(HORIZON):
            v = [d["horizons"][i]["ap"] for d in per_seed.values()]
            cells.append(f"{statistics.mean(v):.4f}")
        print(f"{name:<12}{any_['tf0']:>8.2f}{any_['tf19']:>9.2f}" +
              "".join(f"{c:>9}" for c in cells) + f"{len(per_seed):>7}")
    for metric, label in (("mass_ratio", "masa  sum(p)/burned"), ("area_ratio", "area  p>0.5/burned"),
                          ("f1", "F1@0.5"), ("iou", "IoU@0.5"), ("brier", "Brier")):
        print(f"\n{label:<24}" + "".join(f"{'t+' + str(i + 1):>9}" for i in range(HORIZON)))
        for name, per_seed in res.items():
            cells = []
            for i in range(HORIZON):
                v = [d["horizons"][i].get(metric, float("nan")) for d in per_seed.values()]
                cells.append(f"{statistics.mean(v):.3f}")
            print(f"{name:<24}" + "".join(f"{c:>9}" for c in cells))
    print("\nAP de TEST, pooled, rollout libre (tf=0). El schedule cambia los pesos, asi")
    print("que t+1 tambien puede moverse un poco; el efecto grande esta en t+2..t+5.")
    print("Cada t+k agrupa las muestras cuyo dia t+k existe (menos pixeles al alargar).")
    print("masa/area ~1 y F1/IoU que suben con el AP = mascara mejor; masa/area >> 1")
    print("con F1 plano = el AP premia una prediccion difusa, no una mascara mejor.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--horizons", action="store_true",
                    help="AP de test en t+1..t+5 por schedule (rollout libre)")
    ap.add_argument("--json", type=Path, default=None,
                    help="con --horizons: guardar el resultado")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SUBSTR",
                    help="lanzar solo los trabajos cuyo nombre contenga esto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED_DEFAULT])
    ap.add_argument("--ks", type=float, nargs="+", default=list(KS))
    ap.add_argument("--out", type=Path, default=RUNS / "exp3")
    a = ap.parse_args()

    if a.launch:
        run_queue(matrix(a.out, a.seeds, tuple(a.ks)), a.out / "done",
                  dry_run=a.dry_run, only=a.only)
    if a.table:
        table(a.out, a.seeds, tuple(a.ks))
    if a.horizons:
        res = horizons(a.out, a.seeds, tuple(a.ks), workers=a.workers)
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
            print(f"\n-> {a.json}")
    if not (a.launch or a.table or a.horizons):
        ap.print_help()


if __name__ == "__main__":
    main()
