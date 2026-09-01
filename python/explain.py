"""Explanation generation: LIME (masked or full-frame segmentation) + Grad-CAM.

THE CONCERN-1 EXPERIMENT LIVES HERE. Every LIME config runs under two
segmentation scopes:
  scope='full'  segments the whole frame, background zeros included --
                reproduces the questioned published setup;
  scope='lung'  segments ONLY the lung field (skimage slic mask= / grid cells
                intersected with the lung) -- the reviewer's required recompute.
Both are saved so evaluate.py can print the before/after table the rebuttal needs.

LIME is implemented directly (Bernoulli segment masking + weighted ridge
regression) so the perturbation and kernel match the paper's settings and carry
no external-package drift. Grad-CAM uses forward/backward hooks on the last
conv block (ViT: the last encoder layernorm, reshaped).

Usage (after train.py):
    python explain.py --archs alexnet vgg16 vgg19 resnet50 --seeds 42
    python explain.py --smoke        # 6 images, 100 samples -- pipeline check

Output: pywork/masks/<arch>_s<seed>_<dataset>_<cond>.npz  holding, per image:
    A_<cfg>_<scope>  binary top-K mask       (uint8 HxW)
    SEG_<cfg>_<scope>, IMP_<cfg>_<scope>     segment map + mean importance
    GC               gradcam map (float16)   -- thresholded later at tau
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic

import config as C
from data import load_manifest
from train import build_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ input pipeline
def load_input(dataset: str, cond: str, base: str):
    sub = "roi" if cond == "roi" else "full"
    g = Image.open(C.WORK / dataset / sub / f"{base}.png").convert("L")
    g = g.resize((C.NET_SIZE, C.NET_SIZE), Image.BILINEAR)
    g = np.asarray(g, dtype=np.float32) / 255.0
    m = Image.open(C.WORK / dataset / "mask" / f"{base}.png").convert("L")
    m = np.asarray(m.resize((C.NET_SIZE, C.NET_SIZE), Image.NEAREST)) > 0
    return g, m


def to_tensor(g: np.ndarray) -> torch.Tensor:
    x = np.repeat(g[None], 3, axis=0)
    x = (x - np.array(C.IMAGENET_MEAN)[:, None, None]) / np.array(C.IMAGENET_STD)[:, None, None]
    return torch.from_numpy(x.astype(np.float32))[None].to(DEV)


# ------------------------------------------------------------------ segmentation
def segment(g: np.ndarray, lung: np.ndarray, kind: str, n_seg: int, scope: str):
    """Return int segment map, labels 1..K (0 = outside scope, never ranked)."""
    if kind == "grid":
        side = int(round(np.sqrt(n_seg)))
        ys = np.linspace(0, g.shape[0], side + 1).astype(int)
        xs = np.linspace(0, g.shape[1], side + 1).astype(int)
        seg = np.zeros(g.shape, int)
        k = 1
        for i in range(side):
            for j in range(side):
                seg[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = k
                k += 1
    else:
        mask = lung if scope == "lung" else None
        seg = slic(g, n_segments=n_seg, compactness=0.05, channel_axis=None,
                   mask=mask, start_label=1)
    if scope == "lung":
        # outside-lung pixels get label 0 and are never ranked; a grid cell that
        # straddles the boundary keeps only its in-lung part, so no segment can
        # acquire importance from background zeros (JMI Concern 1).
        seg = seg * lung
    return seg


# ------------------------------------------------------------------ LIME
@torch.no_grad()
def _batch_probs(model, xs, cls):
    p = torch.softmax(model(xs), 1)[:, cls]
    return p.cpu().numpy()


def lime_importance(model, g, seg, cls, n_samples, rng, fill_value):
    """Weighted ridge regression of P(cls) on segment on/off indicators."""
    labels = np.unique(seg[seg > 0])
    K = len(labels)
    if K < 2:
        return labels, np.zeros(len(labels))
    Z = rng.integers(0, 2, size=(n_samples, K)).astype(np.float32)
    Z[0] = 1.0                                            # include the intact image
    probs = np.empty(n_samples, np.float32)
    B = 50
    for s0 in range(0, n_samples, B):
        chunk = Z[s0:s0 + B]
        imgs = np.empty((len(chunk), C.NET_SIZE, C.NET_SIZE), np.float32)
        for i, z in enumerate(chunk):
            gi = g.copy()
            off = labels[z < 0.5]
            if len(off):
                gi[np.isin(seg, off)] = fill_value
            imgs[i] = gi
        x = np.repeat(imgs[:, None], 3, axis=1)
        x = (x - np.array(C.IMAGENET_MEAN)[None, :, None, None]) \
            / np.array(C.IMAGENET_STD)[None, :, None, None]
        probs[s0:s0 + len(chunk)] = _batch_probs(
            model, torch.from_numpy(x.astype(np.float32)).to(DEV), cls)
    d = 1.0 - Z.mean(axis=1)                              # fraction turned off
    w = np.exp(-(d ** 2) / (C.LIME_KERNEL ** 2))
    Zw = Z * w[:, None]
    A = Zw.T @ Z + 1e-3 * np.eye(K)
    b = Zw.T @ probs
    coef = np.linalg.solve(A, b)
    return labels, coef


def topk_mask(seg, labels, coef, K):
    order = np.argsort(coef)[::-1][:min(K, len(labels))]
    return np.isin(seg, labels[order])


# ------------------------------------------------------------------ Grad-CAM
_FEATS = {}


def _hook(mod, inp, out):
    _FEATS["a"] = out
    out.register_hook(lambda gr: _FEATS.__setitem__("g", gr))


def gradcam_layer(model, arch):
    if arch == "alexnet":
        return model.features[-3]
    if arch.startswith("vgg"):
        return model.features[-3]
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "densenet121":
        return model.features[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    if arch == "vit_b_16":
        return model.encoder.layers[-1].ln_1
    raise ValueError(arch)


def gradcam(model, arch, x, cls):
    h = gradcam_layer(model, arch).register_forward_hook(_hook)
    model.zero_grad()
    out = model(x)
    out[0, cls].backward()
    h.remove()
    a, gr = _FEATS["a"], _FEATS["g"]
    if arch == "vit_b_16":                                 # tokens -> grid
        a = a[:, 1:].transpose(1, 2).reshape(1, -1, 14, 14)
        gr = gr[:, 1:].transpose(1, 2).reshape(1, -1, 14, 14)
    wts = gr.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((wts * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(C.NET_SIZE, C.NET_SIZE), mode="bilinear",
                        align_corners=False)[0, 0].detach().cpu().numpy()
    mx = cam.max()
    return cam / mx if mx > 0 else cam


# ------------------------------------------------------------------ driver
def run(arch, seed, dataset, cond, smoke=False):
    ck = torch.load(C.CKPT / f"{arch}_s{seed}.pt", map_location=DEV)
    model = build_model(arch).to(DEV)
    model.load_state_dict(ck["state"])
    model.eval()

    rows = [r for r in load_manifest(dataset) if r["role"] == "heldout"]
    if smoke:
        rows = rows[:6]
    n_samples = 100 if smoke else C.LIME_SAMPLES
    runs = 1 if smoke else C.LIME_RUNS
    store = {}
    print(f"{arch} s{seed} {dataset}/{cond}: {len(rows)} heldout images, "
          f"LIME {n_samples}x{runs}")

    for r_i, row in enumerate(rows):
        base = row["basename"]
        g, lung = load_input(dataset, cond, base)
        x = to_tensor(g)
        with torch.no_grad():
            cls = int(torch.softmax(model(x), 1)[0].argmax())
        fill = float(g.mean())

        for key, kind, n_seg, K in C.LIME_CONFIGS:
            for scope in C.SEG_SCOPES:
                seg = segment(g, lung, kind, n_seg, scope)
                if (seg > 0).sum() == 0:
                    continue
                coefs = []
                for rr in range(runs):
                    rng = np.random.default_rng(C.stable_seed(base, key, scope, rr))
                    labels, coef = lime_importance(model, g, seg, cls,
                                                   n_samples, rng, fill)
                    coefs.append(coef)
                coef = np.mean(coefs, axis=0)
                A = topk_mask(seg, labels, coef, K)
                sfx = f"{key}_{scope}_{base}"
                store[f"A_{sfx}"] = A.astype(np.uint8)
                store[f"SEG_{sfx}"] = seg.astype(np.int16)
                store[f"IMP_{sfx}"] = coef.astype(np.float32)

        store[f"GC_{base}"] = gradcam(model, arch, x, cls).astype(np.float16)
        store[f"CLS_{base}"] = np.array([cls, int(row["label"])], np.int8)
        if (r_i + 1) % 10 == 0:
            print(f"  {r_i+1}/{len(rows)}")

    out = C.MASKSTORE / f"{arch}_s{seed}_{dataset}_{cond}.npz"
    np.savez_compressed(out, **store)
    print(f"[saved] {out}  ({len(store)} arrays)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=C.ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[C.TRAIN_SEEDS[0]])
    ap.add_argument("--dataset", default="shenzhen")
    ap.add_argument("--cond", default="roi", choices=["roi", "full"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    for arch in a.archs:
        for seed in a.seeds:
            run(arch, seed, a.dataset, a.cond, a.smoke)
