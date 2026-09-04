"""Retrain the classifier zoo in PyTorch, multi-seed (JMI Concern 5).

Recipe mirrors the paper's MATLAB training: SGD+momentum, lr 1e-4 backbone /
20x on the fresh head, batch 10, 100 epochs, horizontal flip + small translation,
grayscale replicated to 3 channels, ImageNet-pretrained backbones. The SPLIT is
fixed by DATA_SEED in data.py; TRAIN_SEEDS varies only init/order, so ordering
stability across seeds is measurable.

Usage (GPU box):
    python train.py                          # 4 paper archs x 3 seeds
    python train.py --archs resnet50 --seeds 42 --smoke   # 2-epoch sanity run
    python train.py --archs densenet121 efficientnet_b0 vit_b_16   # third family+

Outputs: pywork/checkpoints/<arch>_s<seed>.pt  and results/train_log.csv
"""
from __future__ import annotations
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import config as C
from data import load_manifest


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RoiDataset(Dataset):
    def __init__(self, dataset: str, role: str, train_aug: bool):
        self.dir = C.WORK / dataset / "roi"
        rows = [r for r in load_manifest(dataset) if r["role"] == role]
        self.items = [(r["basename"], int(r["label"])) for r in rows]
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomAffine(degrees=0, translate=(C.TRANSLATE_FRAC,
                                                             C.TRANSLATE_FRAC))]
        base = [transforms.Resize((C.NET_SIZE, C.NET_SIZE)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)]
        self.tf = transforms.Compose((aug + base) if train_aug else base)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        base, lab = self.items[i]
        img = Image.open(self.dir / f"{base}.png")
        return self.tf(img), lab, base


def build_model(arch: str) -> nn.Module:
    """Pretrained backbone + fresh 2-class head; returns (model, head_params)."""
    fn = getattr(models, arch)
    try:
        m = fn(weights="DEFAULT")
    except Exception as e:                       # offline box: no weight download
        import urllib.error
        if isinstance(e, (urllib.error.URLError, OSError)):
            raise SystemExit(
                f"Cannot download {arch} ImageNet weights (no internet access).\n"
                f"Copy the .pth file for {arch} (URLs in WEIGHTS_URLS.txt) into:\n"
                f"    {C.ROOT / 'torchhub' / 'hub' / 'checkpoints'}\n"
                f"keeping the exact filename, then re-run.") from e
        raise
    if arch == "alexnet" or arch.startswith("vgg"):
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
        head = list(m.classifier[6].parameters())
    elif arch == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, 2)
        head = list(m.fc.parameters())
    elif arch == "densenet121":
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        head = list(m.classifier.parameters())
    elif arch == "efficientnet_b0":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        head = list(m.classifier[1].parameters())
    elif arch == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
        head = list(m.heads.head.parameters())
    else:
        raise ValueError(arch)
    m._head_params = head
    return m


def train_one(arch: str, seed: int, dataset: str, epochs: int, device: str):
    set_seed(seed)
    tr = DataLoader(RoiDataset(dataset, "train", True), batch_size=C.BATCH,
                    shuffle=True, num_workers=2, drop_last=False)
    ho = DataLoader(RoiDataset(dataset, "heldout", False), batch_size=C.BATCH,
                    shuffle=False, num_workers=2)
    model = build_model(arch).to(device)
    head_ids = {id(p) for p in model._head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.SGD([{"params": backbone, "lr": C.LR_BACKBONE},
                           {"params": model._head_params, "lr": C.LR_HEAD}],
                          momentum=C.MOMENTUM)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lossf(model(x), y).backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc, auc = evaluate(model, ho, device)
            print(f"  {arch} s{seed} ep{ep+1:3d}: heldout acc {acc:.3f} auc {auc:.3f} "
                  f"({time.time()-t0:.0f}s)")

    acc, auc = evaluate(model, ho, device)
    ck = C.CKPT / f"{arch}_s{seed}.pt"
    torch.save({"arch": arch, "seed": seed, "state": model.state_dict(),
                "heldout_acc": acc, "heldout_auc": auc, "epochs": epochs}, ck)
    print(f"[saved] {ck}  acc={acc:.3f} auc={auc:.3f}")
    return acc, auc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        p = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
        ps.extend(p.tolist())
        ys.extend(y.numpy().tolist())
    ys, ps = np.array(ys), np.array(ps)
    acc = float(((ps > 0.5).astype(int) == ys).mean())
    # AUC by rank statistic (no sklearn dependency)
    pos, neg = ps[ys == 1], ps[ys == 0]
    if len(pos) and len(neg):
        auc = float((pos[:, None] > neg[None, :]).mean()
                    + 0.5 * (pos[:, None] == neg[None, :]).mean())
    else:
        auc = float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=C.ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=C.TRAIN_SEEDS)
    ap.add_argument("--dataset", default="shenzhen")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, quick sanity")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a checkpoint already exists (overwrites)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  archs={a.archs}  seeds={a.seeds}")
    epochs = 2 if a.smoke else C.EPOCHS

    log = C.RESULTS / "train_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arch", "seed", "epochs", "heldout_acc", "heldout_auc"])
        for arch in a.archs:
            for seed in a.seeds:
                ck = C.CKPT / f"{arch}_s{seed}.pt"
                if ck.is_file() and not a.force and not a.smoke:
                    print(f"[skip] {ck.name} exists (use --force to retrain)")
                    continue
                acc, auc = train_one(arch, seed, a.dataset, epochs, device)
                w.writerow([arch, seed, epochs, f"{acc:.4f}", f"{auc:.4f}"])
                f.flush()
    print(f"log -> {log}")


if __name__ == "__main__":
    main()
