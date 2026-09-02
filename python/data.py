"""Dataset construction: ROI images, masks, and LEAKAGE-FREE splits.

Fixes two review concerns at the data layer, where they belong:
  Concern 2 -- containment was partly measured on training images. Here every
               evaluation set is derived from the SPLIT, and the split is fixed
               by DATA_SEED before any training happens. `manifest.csv` records
               role per image; downstream scripts refuse images whose role
               is not 'heldout'.
  Concern 3 -- OOD accuracy was measured on radiographs whose ROI crops were in
               training. Here the OOD (full-radiograph) set EXCLUDES any basename
               present in the training partition.

Usage (on the GPU box):
    python data.py            # builds pywork/{roi,full,mask}/ + manifest.csv
    python data.py --check    # re-validates an existing manifest (no writes)

Download links: see config.py header.
"""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

import config as C


def _load_gray(p: Path, size: int) -> np.ndarray:
    im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _load_mask(p: Path, size: int) -> np.ndarray:
    im = Image.open(p).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(im) > 0)


# ------------------------------------------------------------ folder auto-detection
_SKIP_DIRS = {"pywork", "venv", ".git", "__pycache__", "checkpoints", "masks", "results"}


def _dirs_under(top: Path, depth: int = 3):
    if not top.is_dir():
        return
    yield top
    if depth == 0:
        return
    for c in sorted(p for p in top.iterdir()
                    if p.is_dir() and p.name not in _SKIP_DIRS):
        yield from _dirs_under(c, depth - 1)


def _count_raw(d: Path, prefix: str) -> int:
    """PNGs named like the dataset, excluding *_mask.png files."""
    return sum(1 for p in d.glob(f"{prefix}_*.png") if not p.stem.endswith("_mask"))


def _find_image_dir(configured: Path, prefix: str, what: str) -> Path:
    """Use the configured folder if present; otherwise search ROOT for the dir
    holding the most <prefix>_*.png files, preferring raw-image folder names."""
    if configured.is_dir() and _count_raw(configured, prefix) > 0:
        return configured
    scored = []
    for d in _dirs_under(C.ROOT):
        n = _count_raw(d, prefix)
        if n == 0:
            continue
        name = d.name.lower()
        bonus = 0
        if "cxr_png" in name:
            bonus += 3
        elif "resized" in name or name in ("cxr", "images", "img", "png"):
            bonus += 2
        # masked/ROI derivatives of the raws also carry these basenames --
        # they must lose to a raw folder whenever one exists
        if any(w in name for w in ("annot", "roi", "mask", "seg")):
            bonus -= 5
        scored.append((bonus, n, d))
    if not scored:
        raise SystemExit(
            f"No {what} images ({prefix}_*.png) found anywhere under {C.ROOT}.\n"
            f"Download links are in the config.py header; or point the "
            f"P2_{what.upper()}_IMG / P2_MONTGOMERY env var at the folder.")
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    bonus, n, d = scored[0]
    print(f"[autodetect] {what} images: {d}  ({n} files)"
          + ("  ** folder name suggests masked/derived images -- verify, or "
             "override with an env var (see config.py) **" if bonus < 0 else ""))
    return d


def _find_mask_dir(configured: Path, prefix: str) -> Path:
    if configured.is_dir() and any(configured.glob(f"{prefix}_*.png")):
        return configured
    for d in _dirs_under(C.ROOT):
        if any(p.stem.endswith("_mask") for p in d.glob(f"{prefix}_*.png")):
            print(f"[autodetect] {prefix} masks: {d}")
            return d
    raise SystemExit(f"No {prefix} lung-mask folder found under {C.ROOT} "
                     "(expected <base>_mask.png files; link in config.py header).")


def _find_named_dir(name: str, prefix: str):
    """Locate e.g. leftMask/rightMask dirs of the NLM Montgomery layout."""
    for d in _dirs_under(C.ROOT, depth=4):
        if d.name.lower() == name.lower() and any(d.glob(f"{prefix}_*.png")):
            return d
    return None


def shenzhen_pairs():
    """Yield (basename, img_path, mask_path, label) for Shenzhen; label 1 = TB."""
    img_dir = _find_image_dir(C.RAW_SHENZHEN_IMG, "CHNCXR", "shenzhen")
    mask_dir = _find_mask_dir(C.RAW_SHENZHEN_MASK, "CHNCXR")
    for img in sorted(img_dir.glob("CHNCXR_*.png")):
        base = img.stem
        if base.endswith("_mask"):
            continue
        m = mask_dir / f"{base}_mask.png"
        if not m.is_file():
            m = mask_dir / f"{base}.png"
        if not m.is_file():
            continue
        yield base, img, m, int(base.endswith("_1"))


def montgomery_pairs():
    """Montgomery: combine left|right manual masks. label 1 = TB."""
    img_dir = _find_image_dir(C.RAW_MONTGOMERY / "CXR_png", "MCUCXR", "montgomery")
    lm = _find_named_dir("leftMask", "MCUCXR")
    rm = _find_named_dir("rightMask", "MCUCXR")
    if lm is None or rm is None:
        raise SystemExit(
            "Montgomery ManualMask/leftMask + rightMask folders not found under "
            f"{C.ROOT}. Montgomery is the optional cross-dataset arm -- download "
            "it from the NLM link in config.py, or skip this step.")
    for img in sorted(img_dir.glob("MCUCXR_*.png")):
        base = img.stem
        lp, rp = lm / f"{base}.png", rm / f"{base}.png"
        if lp.is_file() and rp.is_file():
            yield base, img, (lp, rp), int(base.endswith("_1"))


def build(dataset: str = "shenzhen") -> Path:
    """Write ROI (masked, NOT cropped), full-CXR and mask PNGs + manifest.csv."""
    out_roi = C.WORK / dataset / "roi"
    out_full = C.WORK / dataset / "full"
    out_mask = C.WORK / dataset / "mask"
    for d in (out_roi, out_full, out_mask):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    pairs = list(shenzhen_pairs() if dataset == "shenzhen" else montgomery_pairs())
    if not pairs:
        raise SystemExit(f"No image/mask pairs found for {dataset}. "
                         "Check config paths + download links in config.py.")

    for base, img_p, mask_p, label in pairs:
        g = _load_gray(img_p, C.IMG_SIZE)
        if dataset == "shenzhen":
            m = _load_mask(mask_p, C.IMG_SIZE)
        else:
            lp, rp = mask_p
            m = _load_mask(lp, C.IMG_SIZE) | _load_mask(rp, C.IMG_SIZE)
        if not m.any():
            continue
        roi = g.copy()
        roi[~m] = 0                                        # masking, not cropping
        Image.fromarray(g).save(out_full / f"{base}.png")
        Image.fromarray(roi).save(out_roi / f"{base}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(out_mask / f"{base}.png")
        rows.append({"basename": base, "label": label,
                     "lung_fraction": round(float(m.mean()), 6)})

    # ---- stratified split, fixed by DATA_SEED, BEFORE any training -----------
    rng = random.Random(C.DATA_SEED)
    by_cls = {0: [], 1: []}
    for r in rows:
        by_cls[r["label"]].append(r["basename"])
    heldout = set()
    for lab, names in by_cls.items():
        names = sorted(names)
        rng.shuffle(names)
        k = max(1, round(len(names) * C.SPLIT_HELDOUT))
        heldout.update(names[:k])
    for r in rows:
        r["role"] = "heldout" if r["basename"] in heldout else "train"

    man = C.WORK / dataset / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["basename", "label", "role", "lung_fraction"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["basename"]):
            w.writerow(r)

    n_tr = sum(r["role"] == "train" for r in rows)
    n_ho = sum(r["role"] == "heldout" for r in rows)
    print(f"[{dataset}] {len(rows)} images -> train {n_tr} | heldout {n_ho} "
          f"(DATA_SEED={C.DATA_SEED})")
    print(f"  mean lung fraction: {np.mean([r['lung_fraction'] for r in rows]):.4f}")
    ab = [r["lung_fraction"] for r in rows if r["label"] == 1]
    nm = [r["lung_fraction"] for r in rows if r["label"] == 0]
    print(f"  by class: TB {np.mean(ab):.4f} | normal {np.mean(nm):.4f} "
          "(class-dependent chance level -- report this)")
    print(f"  manifest: {man}")
    return man


def load_manifest(dataset: str = "shenzhen"):
    man = C.WORK / dataset / "manifest.csv"
    with open(man, newline="") as f:
        return list(csv.DictReader(f))


def heldout_basenames(dataset: str = "shenzhen"):
    """The ONLY set containment may be measured on (Concern 2)."""
    return [r["basename"] for r in load_manifest(dataset) if r["role"] == "heldout"]


def ood_clean_basenames(dataset: str = "shenzhen"):
    """Full-radiograph OOD set with training basenames EXCLUDED (Concern 3)."""
    return heldout_basenames(dataset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shenzhen", choices=["shenzhen", "montgomery"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        rows = load_manifest(a.dataset)
        ho = [r for r in rows if r["role"] == "heldout"]
        assert ho, "no heldout rows"
        print(f"manifest OK: {len(rows)} rows, {len(ho)} heldout")
    else:
        build(a.dataset)
