#!/usr/bin/env python3
"""Make the WSTS+ repository run on a current environment. Reversible.

    python wsts+_compat.py --apply     # patch
    python wsts+_compat.py --revert    # back to the published bytes
    python wsts+_compat.py             # show status

NONE of these touch the model, the loss, the hyper-parameters, the data splits
or the evaluation. They are import-level fixes: the repo was written for torch
2.0 and does not import under torch >= 2.1, and its package __init__ pulls in a
module that was never published.
"""
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent / "wsts+"

PATCHES = [
    {
        "file": "src/dataloader/FireSpreadDataset.py",
        "old": "from torch.utils.data.dataset import T_co",
        "new": ("# COMPAT torch>=2.1: T_co is no longer exported from\n"
                "# torch.utils.data.dataset. It is a typing TypeVar and takes\n"
                "# no part in any computation.\n"
                "from typing import TypeVar\n"
                'T_co = TypeVar("T_co", covariant=True)'),
        "why": "torch >= 2.1 stopped exporting T_co; typing only.",
    },
    {
        "file": "src/models/__init__.py",
        "old": "from .TransUnetLightning import TransUnetLightning",
        "new": ("# COMPAT: TransUnetLightning.py is NOT in the public repo, yet\n"
                "# __init__ imports it, so the package does not load as\n"
                "# published. Unused here: this table trains SMPModel.\n"
                "# from .TransUnetLightning import TransUnetLightning"),
        "why": "the imported file does not exist in the published repo.",
    },
    {
        "file": "src/models/__init__.py",
        "old": "from .SegFormerLightning import SegFormerLightning",
        "new": ("# COMPAT: pulls in `transformers`, unused for SMPModel.\n"
                "# from .SegFormerLightning import SegFormerLightning"),
        "why": "drags transformers/timm for a model we do not use.",
    },
    {
        "file": "src/train.py",
        "old": '        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")',
        "new": ('        # COMPAT: the repo was always launched through `wandb agent`,\n'
                '        # which calls wandb.init() in the worker before train.py runs,\n'
                '        # so wandb.run already existed here. Launched standalone,\n'
                "        # Lightning's WandbLogger is lazy and wandb.run is still None.\n"
                '        # Touching .experiment creates the run, exactly as the agent\n'
                '        # did. No effect on the model, the loss or the data.\n'
                '        if wandb.run is None:\n'
                '            _ = self.trainer.logger.experiment\n'
                '        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")  # COMPAT-ANCHOR'),
        "why": "wandb.run is None unless launched by a wandb sweep agent.",
    },
    {
        "file": "src/models/BaseModel.py",
        "old": "                alpha=1 - self.hparams.pos_class_weight,",
        "new": ('                # COMPAT: this alpha is NEGATIVE in every published run.\n'
                '                # BaseModel.__init__ normalises pos_class_weight into\n'
                '                # (0,1), but the subclass calls save_hyperparameters()\n'
                '                # after super().__init__(), which restores the raw value\n'
                '                # (236, or whatever before_instantiate_classes computed\n'
                '                # from the fold). So alpha = 1 - 236 = -235.\n'
                '                # torchvision 0.15.1, pinned in requirements.txt, applies\n'
                '                # the class re-weighting only `if alpha >= 0`, so it was\n'
                '                # silently skipped: the published numbers come from focal\n'
                '                # loss with gamma=2 and NO class weighting. Current\n'
                '                # torchvision raises ValueError instead. Passing -1, its\n'
                '                # documented "no alpha" value, reproduces the published\n'
                '                # behaviour exactly. Do NOT "fix" this to weight the\n'
                '                # positive class: that would be a different model from\n'
                '                # the one their AP numbers were measured on.\n'
                '                alpha=min(1 - self.hparams.pos_class_weight, 1.0)\n'
                '                if 1 - self.hparams.pos_class_weight >= 0\n'
                '                else -1.0,  # COMPAT-ALPHA'),
        "why": "alpha < 0: torchvision 0.15 skipped it, current one raises.",
    },
    {
        "file": "src/models/SMPTempModel.py",
        "old": "        decoder_output = self.model.decoder(*decoder_features)",
        "new": ('        # COMPAT: segmentation-models-pytorch changed this signature.\n'
                '        # 0.3.2, pinned in requirements.txt, had\n'
                '        #     UnetDecoder.forward(self, *features)\n'
                '        # and 0.4+ takes a single list:\n'
                '        #     UnetDecoder.forward(self, features: List[Tensor])\n'
                '        # Same tensors, same order -- only the call convention\n'
                '        # differs. SMPTempModel is the only place this bites,\n'
                '        # because it drives the decoder by hand; SMPModel calls\n'
                '        # smp.Unet as a whole and is unaffected.\n'
                '        # NOTE the newer decoder upsamples to the recorded spatial\n'
                '        # shape instead of scale_factor=2. Identical for the\n'
                '        # power-of-two crops used here (64, 128); it would differ\n'
                '        # for odd input sizes.\n'
                '        decoder_output = self.model.decoder(decoder_features)  # COMPAT-SMP'),
        "why": "smp >= 0.4 decoder takes a list, not *args.",
    },
    {
        "file": "src/dataloader/FireSpreadDataset.py",
        "old": '        for fire_year in self.datapoints_per_fire:    \n            for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():\n                if target_id - first_id_in_current_fire < datapoints_in_fire:\n                    found_fire_year = fire_year\n                    found_fire_name = fire_name\n                    break\n                else:\n                    first_id_in_current_fire += datapoints_in_fire',
        "new": "        # COMPAT (a BUG, not a version issue): the published loop has no\n        # guard after the inner break, so the outer loop keeps scanning\n        # later years and overwrites the fire it just found. Measured on\n        # fold 0's four training years: all 401 sampled indices resolve\n        # to 2021, the last year, and 272 of them name the wrong fire --\n        # the 7,071 datapoints of 2016/2017/2020 are unreachable. The\n        # research copy of this same file (in\n        # model/experiments_wildfirespreadts/official_repo) already\n        # carries this guard, and that is the version our published\n        # anchors were measured with. Without it, any WSTS+ baseline is\n        # trained and scored on one mangled year.\n        for fire_year in self.datapoints_per_fire:\n            if found_fire_year is None:  # COMPAT-INDEX\n                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():\n                    if target_id - first_id_in_current_fire < datapoints_in_fire:\n                        found_fire_year = fire_year\n                        found_fire_name = fire_name\n                        break\n                    else:\n                        first_id_in_current_fire += datapoints_in_fire",
        "why": "index lookup never stops: every sample resolves to the last year.",
    },
]


def status():
    for p in PATCHES:
        f = REPO / p["file"]
        if not f.exists():
            print("  FALTA    %s" % p["file"]); continue
        s = f.read_text()
        state = ("parcheado" if is_patched(p, s)
                 else "original" if p["old"] in s else "?")
        print("  %-10s %-44s %s" % (state, p["file"], p["why"]))


def is_patched(p, s):
    return p["new"].splitlines()[-1] in s


def apply(revert=False):
    for p in PATCHES:
        f = REPO / p["file"]
        s = f.read_text()
        # Idempotent: applying twice used to nest the COMPAT comment inside
        # itself, because `old` survives as a substring of `new`.
        if is_patched(p, s) != revert:
            print("  sin cambios %s (%s)" % (p["file"], p["why"]))
            continue
        a, b = (p["new"], p["old"]) if revert else (p["old"], p["new"])
        if a in s:
            f.write_text(s.replace(a, b))
            print("  %s %s" % ("revertido" if revert else "parcheado", p["file"]))
        else:
            print("  sin cambios %s" % p["file"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.apply:
        apply()
    elif a.revert:
        apply(revert=True)
    else:
        status()
