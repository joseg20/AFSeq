#!/usr/bin/env python3
"""Exp4 -- TEST scoring of the fused checkpoints, one column per benchmark.

    python experiments/exp4/eval_test.py --models utae_ar --sources afseq

Same protocol as exp2's table: pooled AP at t+1, crop 128, valid mask on t+1,
ties collapsed as sklearn (evaluation/metrics.py). Each checkpoint is scored on
each benchmark's own test set, in its own column -- never pooled (see exp4.py).
The AFSeq test set is fold-independent; the WSTS+ test years change per fold.

Only utae_ar for now: the "theirs" rows (utae_res18, res50_unet) come out of
train_theirs.py with their own checkpoint format and need their own loader.

Results append into RUNS/exp4/eval_test.json, one entry per (model, source,
fold, benchmark), so partial invocations while the matrix still trains are fine.
"""
from __future__ import annotations

import argparse
import json, os
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "training", ROOT / "benchmarks",
           ROOT / "experiments", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from paths import RUNS                                          # noqa: E402
from evaluation.metrics import pooled_ap_gpu                    # noqa: E402
from train import SINGLE_STEP, build_model, forward             # noqa: E402
from fusion import N_VEG, collate, single                       # noqa: E402
from exp4 import tag_of                                         # noqa: E402


def load_res18(tag: str, out_dir: Path):
    """Modelo res18 desde el JSON del run (original o clon en otro disco).

    El JSON dice donde esta el best_ckpt; los clones lo escriben con la ruta
    vista desde el otro nodo, que aqui cuelga de otro punto de montaje.
    Los centinelas (placeholder) no cuentan como runs hechos.
    """
    import json as _json
    for d in (out_dir, out_dir.parent / "exp4_mirror"):
        js = d / f"{tag}.json"
        if not js.exists():
            continue
        meta = _json.loads(js.read_text())
        if "placeholder" in meta or not meta.get("best_ckpt"):
            continue
        # CKPT_REWRITE="viejo=nuevo" reescribe el prefijo del checkpoint cuando
        # el run se lanzo con otro punto de montaje.
        ck = meta["best_ckpt"]
        rw = os.environ.get("CKPT_REWRITE", "")
        if "=" in rw:
            a_, b_ = rw.split("=", 1)
            ck = ck.replace(a_, b_)
        if not Path(ck).exists():
            continue
        sys.path.insert(0, str(ROOT / "benchmarks" / "wsts+"))
        from train_theirs import build as build_theirs
        m = build_theirs("utae_res18")
        m.load_state_dict(torch.load(ck, map_location="cpu",
                                     weights_only=False)["state_dict"])
        return m, ck
    return None, None


def loader_of(benchmark: str, fold: int, batch_size: int, workers: int):
    # horizon=1: this scores t+1 only, and the 5-step checkpoint loads into a
    # 1-step model unchanged (same trick as exp2's table)
    ds = single(benchmark, fold, "test", horizon=1)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      collate_fn=collate)


@torch.no_grad()
def score(model, name, loader, dev, limit=0):
    ys, ps = [], []
    for i, (X, FC, Y, V, _) in enumerate(loader):
        if limit and i >= limit:
            break
        X, FC = X.to(dev, non_blocking=True), FC.to(dev, non_blocking=True)
        Y, V = Y.to(dev, non_blocking=True), V.to(dev, non_blocking=True)
        keep = V[:, 0] > 0.5
        if not keep.any():
            continue
        if name == "utae_res18":
            p = torch.sigmoid(model(X)[:, 0])
        else:
            out = forward(model, name, X, FC, None, 0.0)
            p = (torch.sigmoid(out[:, 0]) if name in SINGLE_STEP
                 else torch.sigmoid(out[:, 0, 0]))
        ys.append((Y[keep, 0, 0] > 0.5).flatten())
        ps.append(p[keep].flatten())
    if not ys:
        return float("nan"), 0
    n = sum(int(y.numel()) for y in ys)
    return pooled_ap_gpu(torch.cat(ys), torch.cat(ps), device=dev), n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["utae_ar"])
    ap.add_argument("--sources", nargs="+", default=["wstsplus", "afseq", "both"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="cortar tras N lotes por split; solo smoke test")
    ap.add_argument("--anchor-dir", type=Path, default=None,
                    help="directorio de best_AR_Veg_p4f*.pth para la fuente "
                         "pseudo 'wsts_anchor'")
    ap.add_argument("--out", type=Path, default=RUNS / "exp4")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_path = a.out / "eval_test.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}

    # the AFSeq test split never changes with the fold: build its loader once
    afseq_loader = loader_of("afseq", 0, a.batch_size, a.workers)

    for model_name in a.models:
        if model_name not in ("utae_ar", "utae_res18"):
            print(f"{model_name}: sin soporte todavia")
            continue
        for source in a.sources:
            for fold in a.folds:
                if model_name == "utae_res18":
                    tag = tag_of(model_name, source, fold)
                    m, ck = load_res18(tag, a.out)
                    if m is None:
                        print(f"  {tag}: sin checkpoint todavia")
                        continue
                    m = m.to(dev).eval()
                    for bench, ldr in (("afseq", afseq_loader),
                                       ("wstsplus", loader_of("wstsplus", fold,
                                                              a.batch_size,
                                                              a.workers))):
                        ap_, n = score(m, model_name, ldr, dev, a.limit)
                        key = f"{model_name}|{source}|f{fold}|{bench}"
                        res[key] = dict(ap=ap_, pixels=n, ckpt=str(ck))
                        print(f"  {tag} on {bench:9s} testAP={ap_:.4f} "
                              f"({n:,} px)", flush=True)
                        if not a.limit:
                            res_path.write_text(json.dumps(res, indent=1))
                    del m
                    torch.cuda.empty_cache()
                    continue
                if source == "wsts_anchor":
                    # published anchor checkpoints (trained on WSTS+ Veg, the
                    # ones behind exp1's 0.4299 reproduction), scored cross
                    ck = (Path(a.anchor_dir)
                          / f"best_AR_Veg_p4f{fold}_bce_dice_lr0p001.pth")
                    tag = ck.stem
                else:
                    tag = tag_of(model_name, source, fold)
                    ck = a.out / f"best_{tag}.pth"
                if not ck.exists():
                    print(f"  {tag}: sin checkpoint todavia")
                    continue
                m = build_model(model_name, 1, in_ch=N_VEG)
                m.load_state_dict(torch.load(ck, map_location="cpu"))
                m = m.to(dev).eval()
                for bench, ldr in (("afseq", afseq_loader),
                                   ("wstsplus", loader_of("wstsplus", fold,
                                                          a.batch_size,
                                                          a.workers))):
                    ap_, n = score(m, model_name, ldr, dev, a.limit)
                    key = f"{model_name}|{source}|f{fold}|{bench}"
                    res[key] = dict(ap=ap_, pixels=n)
                    print(f"  {tag} on {bench:9s} testAP={ap_:.4f} ({n:,} px)",
                          flush=True)
                    if not a.limit:
                        res_path.write_text(json.dumps(res, indent=1))
                del m
                torch.cuda.empty_cache()

    # summary: mean +- std over folds, per (source, benchmark column)
    print(f"\n{'trained on':<12}{'model':<10}{'AFSeq test':>16}{'WSTS+ test':>16}")
    for source in a.sources:
        for model_name in a.models:
            cells = []
            for bench in ("afseq", "wstsplus"):
                vals = [res[k]["ap"] for f in a.folds
                        if (k := f"{model_name}|{source}|f{f}|{bench}") in res]
                cells.append(f"{statistics.mean(vals):.4f}+-"
                             f"{statistics.stdev(vals):.4f}" if len(vals) > 1
                             else (f"{vals[0]:.4f}" if vals else "-"))
            print(f"{source:<12}{model_name:<10}{cells[0]:>16}{cells[1]:>16}")
    print("\nNota: AP pooled a t+1, test de cada benchmark por separado.")


if __name__ == "__main__":
    main()
