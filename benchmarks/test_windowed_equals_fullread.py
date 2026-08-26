"""Prueba: la lectura por ventana es BIT-IDENTICA al camino original.

Mismo seed de np.random antes de cada muestra -> mismos draws -> mismo crop,
mismos flips; todo lo demas es por-pixel. Se comparan los cinco campos del
sample con torch.equal en train (aleatorio) y test (determinista).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wstsplus_data import make_dataset  # noqa: E402

DATA = os.environ.get("WSTS_HDF5") or sys.exit("define WSTS_HDF5")
fallos = 0
for part in ("test", "train"):
    for feats in ("All", "Veg"):
        ds = make_dataset(0, part, DATA, features=feats, history="fixed",
                          horizon=5, crop=128)
        idxs = np.linspace(0, len(ds) - 1, 8, dtype=int)
        for i in idxs:
            np.random.seed(7000 + int(i))
            os.environ.pop("WSTS_FULLREAD", None)
            a = ds[int(i)]
            np.random.seed(7000 + int(i))
            os.environ["WSTS_FULLREAD"] = "1"
            b = ds[int(i)]
            os.environ.pop("WSTS_FULLREAD", None)
            iguales = all(torch.equal(x, y) if torch.is_tensor(x) else x == y
                          for x, y in zip(a, b))
            if not iguales:
                fallos += 1
                difs = [k for k, (x, y) in enumerate(zip(a, b))
                        if torch.is_tensor(x) and not torch.equal(x, y)]
                print(f"DIFIERE {part}/{feats} idx={i} campos={difs}")
        print(f"{part}/{feats}: {len(idxs)} muestras comparadas")
print("IDENTICO" if fallos == 0 else f"FALLOS={fallos}")
