"""Containment + nulls + seed-stability evaluation (the revision's number source).

What this produces, and which review concern each answers:
  containment.csv      lift over chance, HELD-OUT IMAGES ONLY        (Concern 2)
                       for LIME under BOTH segmentation scopes       (Concern 1)
                       and Grad-CAM at tau.
  nulls.csv            rotation, translation AND mask-swap nulls per (Concern 6/7)
                       config, with negative-lift interpretation.
  seed_stability.csv   architecture orderings per seed + pairwise    (Concern 5)
                       Kendall tau between seeds and between explainers.
  scope_delta.csv      the Concern-1 before/after table: full-frame vs
                       lung-restricted segmentation, side by side.

Pure numpy/scipy on the saved mask store -- no GPU, minutes.

Usage:
    python evaluate.py --archs alexnet vgg16 vgg19 resnet50 --seeds 42 43 44
"""
from __future__ import annotations
import argparse
import csv
import itertools

import numpy as np

import config as C
from containment import precision, reference_fraction, lift_analytic
from nulls import lift_geometry, lift_maskswap, interpret_negative_lift
from explain import load_input
from data import load_manifest


def kendall(a, b):
    """Kendall tau over paired rankings (small n; descriptive only)."""
    n = len(a)
    c = d = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            c += s > 0
            d += s < 0
    return (c - d) / (c + d) if (c + d) else float("nan")


def load_store(arch, seed, dataset, cond):
    p = C.MASKSTORE / f"{arch}_s{seed}_{dataset}_{cond}.npz"
    return np.load(p) if p.is_file() else None


def per_image_lifts(z, dataset, cond, key, scope, lung_masks):
    """Return dict base -> (A, B, lift_analytic) for one config/scope."""
    out = {}
    for k in z.files:
        pre = f"A_{key}_{scope}_"
        if k.startswith(pre):
            base = k[len(pre):]
            A = z[k].astype(bool)
            B = lung_masks[base]
            if A.any() and B.any():
                out[base] = (A, B)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=C.ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=C.TRAIN_SEEDS)
    ap.add_argument("--dataset", default="shenzhen")
    ap.add_argument("--cond", default="roi")
    a = ap.parse_args()

    heldout = {r["basename"] for r in load_manifest(a.dataset) if r["role"] == "heldout"}
    lung_masks = {}
    for base in sorted(heldout):
        _, m = load_input(a.dataset, a.cond, base)
        lung_masks[base] = m
    all_B = list(lung_masks.values())

    cont_rows, null_rows = [], []
    # per (arch,seed,config,scope) mean lift -- reused for stability + scope delta
    mean_lift = {}

    for arch in a.archs:
        for seed in a.seeds:
            z = load_store(arch, seed, a.dataset, a.cond)
            if z is None:
                print(f"[skip] no mask store for {arch} s{seed}")
                continue

            # ---------- LIME configs x scopes ----------
            for key, _, _, K in C.LIME_CONFIGS:
                for scope in C.SEG_SCOPES:
                    pairs = per_image_lifts(z, a.dataset, a.cond, key, scope, lung_masks)
                    pairs = {b: v for b, v in pairs.items() if b in heldout}
                    if not pairs:
                        continue
                    la, lg, lt, lsw, negs = [], [], [], [], 0
                    for b, (A, B) in sorted(pairs.items()):
                        rng = np.random.default_rng(C.stable_seed(arch, seed, key, scope, b))
                        la.append(lift_analytic(A, B))
                        g_l, _, _ = lift_geometry(A, B, C.NULL_DRAWS, rng, "rotation")
                        t_l, _, _ = lift_geometry(A, B, C.NULL_DRAWS, rng, "translation")
                        others = [m for bb, m in lung_masks.items() if bb != b]
                        s_l, _, _ = lift_maskswap(A, B, others, C.NULL_DRAWS, rng)
                        lg.append(g_l); lt.append(t_l); lsw.append(s_l)
                        negs += g_l < -0.02
                    mean_lift[(arch, seed, key, scope)] = float(np.mean(la))
                    cont_rows.append([arch, seed, key, scope, len(la),
                                      round(float(np.mean(la)), 4),
                                      round(float(np.std(la) / np.sqrt(len(la))), 4)])
                    null_rows.append([arch, seed, key, scope, len(la),
                                      round(float(np.mean(la)), 4),
                                      round(float(np.mean(lg)), 4),
                                      round(float(np.mean(lt)), 4),
                                      round(float(np.mean(lsw)), 4),
                                      negs,
                                      interpret_negative_lift(float(np.mean(lg)))])

            # ---------- Grad-CAM ----------
            gc_la, gc_lg, gc_sw = [], [], []
            for b in sorted(heldout):
                kk = f"GC_{b}"
                if kk not in z.files:
                    continue
                A = z[kk].astype(np.float32) >= C.GRADCAM_TAU
                B = lung_masks[b]
                if not A.any():
                    continue
                rng = np.random.default_rng(C.stable_seed(arch, seed, "gc", b))
                gc_la.append(lift_analytic(A, B))
                gl, _, _ = lift_geometry(A, B, C.NULL_DRAWS, rng, "rotation")
                sl, _, _ = lift_maskswap(A, B, [m for bb, m in lung_masks.items()
                                                if bb != b], C.NULL_DRAWS, rng)
                gc_lg.append(gl); gc_sw.append(sl)
            if gc_la:
                mean_lift[(arch, seed, "GradCAM", "-")] = float(np.mean(gc_la))
                cont_rows.append([arch, seed, "GradCAM@%.1f" % C.GRADCAM_TAU, "-",
                                  len(gc_la), round(float(np.mean(gc_la)), 4),
                                  round(float(np.std(gc_la) / np.sqrt(len(gc_la))), 4)])
                null_rows.append([arch, seed, "GradCAM@%.1f" % C.GRADCAM_TAU, "-",
                                  len(gc_la), round(float(np.mean(gc_la)), 4),
                                  round(float(np.mean(gc_lg)), 4), "",
                                  round(float(np.mean(gc_sw)), 4),
                                  int(sum(x < -0.02 for x in gc_lg)),
                                  interpret_negative_lift(float(np.mean(gc_lg)))])
            print(f"[done] {arch} s{seed}")

    # ---------------- write ----------------
    def w(path, header, rows):
        with open(path, "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)
        print(f"-> {path}  ({len(rows)} rows)")

    w(C.RESULTS / "containment.csv",
      ["arch", "seed", "config", "scope", "n", "lift_analytic", "se"], cont_rows)
    w(C.RESULTS / "nulls.csv",
      ["arch", "seed", "config", "scope", "n", "lift_analytic", "lift_rotation",
       "lift_translation", "lift_maskswap", "n_negative_geom", "interpretation"],
      null_rows)

    # ---------------- Concern-1 before/after ----------------
    sd = []
    for (arch, seed, key, scope), v in sorted(mean_lift.items()):
        if scope == "full":
            v2 = mean_lift.get((arch, seed, key, "lung"))
            if v2 is not None:
                sd.append([arch, seed, key, round(v, 4), round(v2, 4),
                           round(v2 - v, 4)])
    w(C.RESULTS / "scope_delta.csv",
      ["arch", "seed", "config", "lift_fullframe", "lift_lungscoped", "delta"], sd)

    # ---------------- seed stability + explainer agreement ----------------
    st = []
    for key, _, _, _ in C.LIME_CONFIGS:
        for scope in C.SEG_SCOPES:
            for s1, s2 in itertools.combinations(a.seeds, 2):
                r1 = [mean_lift.get((arch, s1, key, scope)) for arch in a.archs]
                r2 = [mean_lift.get((arch, s2, key, scope)) for arch in a.archs]
                if None in r1 or None in r2:
                    continue
                st.append([key, scope, f"seed{s1}-vs-seed{s2}",
                           round(kendall(r1, r2), 3),
                           " < ".join(x for _, x in sorted(zip(r1, a.archs)))])
    for scope in C.SEG_SCOPES:
        for seed in a.seeds:
            lime = [mean_lift.get((arch, seed, "LIME-fine-m100-K20", scope))
                    for arch in a.archs]
            gc = [mean_lift.get((arch, seed, "GradCAM", "-")) for arch in a.archs]
            if None in lime or None in gc:
                continue
            st.append([f"LIMEfine-vs-GradCAM", scope, f"seed{seed}",
                       round(kendall(lime, gc), 3),
                       "lime:" + "<".join(x for _, x in sorted(zip(lime, a.archs)))
                       + " | gc:" + "<".join(x for _, x in sorted(zip(gc, a.archs)))])
    w(C.RESULTS / "seed_stability.csv",
      ["comparison", "scope", "pair", "kendall_tau", "orderings"], st)

    print("\nRead scope_delta.csv first: if lung-scoped lifts differ materially from "
          "full-frame, Concern 1 is confirmed and the paper's finding shifts to the "
          "masked-input interaction -- which the reviewer called 'itself publishable'.")


if __name__ == "__main__":
    main()
