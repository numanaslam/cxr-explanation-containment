# Python revision pipeline — cxr-explanation-containment

Full PyTorch reimplementation of the paper's pipeline, built to answer the JMI
review point by point. Replaces the MATLAB dependency; models are **retrained**
(no ONNX bridge), so all numbers regenerate self-consistently.

| Stage | Script | Review concern it answers |
|---|---|---|
| 0 | `test_core.py` | correctness of the metric/null core (runs anywhere, no GPU) |
| 1 | `data.py` | **C2** leakage-free split fixed before training; **C3** clean OOD set |
| 2 | `train.py` | **C5** multi-seed retraining; optional 3rd family (DenseNet/EfficientNet/ViT) |
| 3 | `explain.py` | **C1** LIME under *full-frame* AND *lung-restricted* segmentation; Grad-CAM |
| 4 | `occlusion.py` | **C1b** admissibility gate under both fills, both scopes |
| 5 | `evaluate.py` | containment (held-out only), rotation/translation/**mask-swap** nulls (**C6/7**), seed stability + Kendall τ (**C5**), and `scope_delta.csv` — the C1 before/after table |

Dataset download links: header of `config.py`.

## Setup (GPU box, once)

Windows PowerShell (note: PS 5.1 has no `&&` — run lines one at a time):

```powershell
cd C:\paper2_py
python -m venv venv
.\venv\Scripts\Activate.ps1
# if activation is blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
# CUDA torch FIRST (so pip does not grab the CPU wheel), then the rest:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Linux/macOS:

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision   # pick the CUDA index for your driver from pytorch.org
pip install -r requirements.txt
```

Paths: everything is rooted at `P2_ROOT` (default `C:\paper2_repo`). Either lay the
raw data out as `config.py` expects, or edit the `RAW_*` constants there:

```
%P2_ROOT%\input\CXR_png\CHNCXR_####_[01].png      Shenzhen images
%P2_ROOT%\input\mask\<base>_mask.png              Shenzhen manual lung masks
%P2_ROOT%\input\MontgomerySet\...                 NLM Montgomery layout
```

Everything the pipeline creates lives under `%P2_ROOT%\pywork\` — raw data is
never written to.

## Run order

```bash
# --- 0. sanity: metric/null core (no data, no GPU, seconds)
python test_core.py

# --- 1. build ROI/full/mask sets + the fixed, leakage-free split
python data.py --dataset shenzhen
python data.py --dataset montgomery          # optional cross-dataset arm

# --- 2. SMOKE TEST the whole chain first (~15 min total)
python train.py   --archs resnet50 --seeds 42 --smoke
python explain.py --archs resnet50 --seeds 42 --smoke
python occlusion.py --archs resnet50 --seeds 42 --smoke
python evaluate.py --archs resnet50 --seeds 42

# --- 3. full training: 4 archs x 3 seeds (the long stage; ~overnight)
python train.py

# --- 4. explanations on the HELD-OUT split, both conditions
python explain.py --seeds 42 43 44 --cond roi
python explain.py --seeds 42 43 44 --cond full     # OOD arm (clean set)

# --- 5. admissibility gate (minutes)
python occlusion.py --seeds 42 43 44

# --- 6. all evaluation CSVs (CPU, minutes)
python evaluate.py --seeds 42 43 44

# --- 7. optional third architecture family (Concern 5)
python train.py   --archs densenet121 efficientnet_b0 vit_b_16
python explain.py --archs densenet121 efficientnet_b0 vit_b_16 --seeds 42 43 44
python evaluate.py --archs alexnet vgg16 vgg19 resnet50 densenet121 efficientnet_b0 vit_b_16 --seeds 42 43 44
```

## Reading the results (in this order)

1. **`results/scope_delta.csv`** — the decision table. If lung-restricted lifts
   differ materially from full-frame, the review's Concern 1 is confirmed and the
   paper's finding becomes the masked-input × perturbation-explainer interaction
   (which the editor called "itself publishable, and arguably more useful").
2. `results/nulls.csv` — does rotation-vs-analytic shrinkage survive the lung
   scope? Do rotation and mask-swap nulls agree (if yes, the conclusion is not
   null-specific)? Negative geometry lifts come pre-interpreted.
3. `results/seed_stability.csv` — is the architecture ordering stable across
   seeds? If τ between seeds is low, the ordering claim is retired, per review.
4. `results/occlusion_gate.csv` — does the robust/fragile split survive the lung
   scope and agree across fills? If the gate flips with fill value, it is not yet
   an admissibility gate and the paper must say so.
5. `results/containment.csv`, `results/train_log.csv` — the raw tables.

## Notes

- The train/held-out split is fixed by `DATA_SEED=42` in `config.py` and never
  varies; `TRAIN_SEEDS` varies only initialisation/order. Do not change
  `DATA_SEED` between stages — every stage keys off `manifest.csv`.
- All randomness is derived from explicit seeds or stable per-image hashes, so
  every stage is resumable and reproducible.
- `--smoke` runs 6 images / 100 LIME samples / 2 epochs; use it after any edit.
