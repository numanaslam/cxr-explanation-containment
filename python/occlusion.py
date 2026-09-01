"""Occlusion-robustness admissibility gate (JMI Concern 1b: fill dependence).

For each image/config: drop in predicted-class probability when occluding
  top-K segments | area-matched random segments | bottom-K segments
under BOTH fills ('mean', 'zero'), plus importance concentration. The review's
point -- that the gate's output moved with the fill value for some
architectures -- is met by reporting both fills side by side, always, and by
computing the gate on lung-scoped segmentations where background occlusion
cannot masquerade as fragility.

Reads the mask store written by explain.py; needs only ~15 forward passes per
image/config, so minutes on GPU.

Usage:
    python occlusion.py --archs alexnet vgg16 vgg19 resnet50 --seeds 42
Output: results/occlusion_gate.csv
"""
from __future__ import annotations
import argparse
import csv

import numpy as np
import torch

import config as C
from explain import load_input, to_tensor, DEV
from train import build_model
from data import load_manifest


@torch.no_grad()
def prob_of(model, g, cls):
    return float(torch.softmax(model(to_tensor(g)), 1)[0, cls])


def occlude(g, seg, labels_sel, fill):
    gi = g.copy()
    gi[np.isin(seg, labels_sel)] = fill
    return gi


def run(arch, seed, dataset, cond, cfg_keys, smoke=False):
    ck = torch.load(C.CKPT / f"{arch}_s{seed}.pt", map_location=DEV)
    model = build_model(arch).to(DEV)
    model.load_state_dict(ck["state"])
    model.eval()

    z = np.load(C.MASKSTORE / f"{arch}_s{seed}_{dataset}_{cond}.npz")
    rows_out = []
    bases = sorted({k[len("CLS_"):] for k in z.files if k.startswith("CLS_")})
    if smoke:
        bases = bases[:6]

    for key in cfg_keys:
        Kc = int(key.rsplit("K", 1)[1])
        for scope in C.SEG_SCOPES:
            acc = {f: {"top": [], "rnd": [], "bot": []} for f in C.OCC_FILLS}
            conc = []
            for base in bases:
                sk = f"{key}_{scope}_{base}"
                if f"SEG_{sk}" not in z.files:
                    continue
                seg = z[f"SEG_{sk}"]
                imp = z[f"IMP_{sk}"]
                labels = np.unique(seg[seg > 0])
                if len(labels) < 3:
                    continue
                g, _ = load_input(dataset, cond, base)
                cls = int(z[f"CLS_{base}"][0])
                p0 = prob_of(model, g, cls)

                order = np.argsort(imp)[::-1]
                K = min(Kc, len(labels))
                top = labels[order[:K]]
                bot = labels[order[-K:]]
                area_t = np.isin(seg, top).sum()
                conc.append(float(np.abs(imp[order[:K]]).sum()
                                  / max(np.abs(imp).sum(), 1e-9) - K / len(labels)))

                rng = np.random.default_rng(C.stable_seed(base, key, scope))
                rand_sets = []
                for _ in range(C.OCC_RAND_DRAWS):
                    perm = rng.permutation(labels)
                    sel, area = [], 0
                    for lab in perm:
                        sel.append(lab)
                        area += (seg == lab).sum()
                        if area >= area_t:
                            break
                    rand_sets.append(np.array(sel))

                for f in C.OCC_FILLS:
                    fill = float(g.mean()) if f == "mean" else 0.0
                    acc[f]["top"].append(p0 - prob_of(model, occlude(g, seg, top, fill), cls))
                    acc[f]["bot"].append(p0 - prob_of(model, occlude(g, seg, bot, fill), cls))
                    dr = [p0 - prob_of(model, occlude(g, seg, s, fill), cls)
                          for s in rand_sets]
                    acc[f]["rnd"].append(float(np.mean(dr)))

            for f in C.OCC_FILLS:
                if not acc[f]["top"]:
                    continue
                dt, dr, db = (np.array(acc[f][k]) for k in ("top", "rnd", "bot"))
                gap = dt - dr
                rows_out.append([arch, seed, dataset, cond, key, scope, f, len(dt),
                                 round(float(dt.mean()), 4), round(float(dr.mean()), 4),
                                 round(float(db.mean()), 4), round(float(gap.mean()), 4),
                                 round(float(np.mean(conc)), 4)])
                print(f"{arch} s{seed} {key}/{scope} fill={f}: "
                      f"dTop={dt.mean():.3f} dRAND={dr.mean():.3f} "
                      f"dBot={db.mean():.3f} gap={gap.mean():+.3f}")
    return rows_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=C.ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[C.TRAIN_SEEDS[0]])
    ap.add_argument("--dataset", default="shenzhen")
    ap.add_argument("--cond", default="roi")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfgs = [k for k, _, _, _ in C.LIME_CONFIGS]
    out = C.RESULTS / "occlusion_gate.csv"
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["arch", "seed", "dataset", "cond", "config", "scope", "fill",
                        "n", "drop_topk", "drop_random", "drop_bottomk", "gap",
                        "concentration_excess"])
        for arch in a.archs:
            for seed in a.seeds:
                for row in run(arch, seed, a.dataset, a.cond, cfgs, a.smoke):
                    w.writerow(row)
                fh.flush()
    print(f"-> {out}")
