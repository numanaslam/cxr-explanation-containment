"""Central configuration: paths, seeds, recipes, and dataset download links.

Everything downstream imports from here, so path/layout changes happen once.

DATA DOWNLOAD LINKS (primary sources; cite these, not mirrors)
--------------------------------------------------------------
Shenzhen + Montgomery TB chest X-rays (NLM official):
    https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
Shenzhen manual lung masks (Kaggle annotation set):
    https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities
Montgomery manual lung masks ship inside the NLM download (ManualMask/left|rightMask).

Lesion-level priors (for the lesion-prior experiment -- JMI Concern 1+5):
    NIH ChestX-ray14 + BBox_List_2017.csv : https://nihcc.app.box.com/v/ChestXray-NIHCC
        (Kaggle mirror: https://www.kaggle.com/datasets/nih-chest-xrays/data)
    RSNA pneumonia boxes : https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
    VinDr-CXR            : https://physionet.org/content/vindr-cxr/
    CheXlocalize         : https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c

Analysis code: https://github.com/numanaslam/cxr-explanation-containment
"""
from __future__ import annotations
import os
import zlib
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("P2_ROOT", r"C:\paper2_repo")).expanduser()


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else default


# Preferred layout; when a folder is missing, data.py auto-detects the real one
# under ROOT (bounded search) and prints what it picked. Env vars override both.
RAW_SHENZHEN_IMG = _env_path("P2_SHENZHEN_IMG", ROOT / "input" / "CXR_png")
RAW_SHENZHEN_MASK = _env_path("P2_SHENZHEN_MASK", ROOT / "input" / "mask")
RAW_MONTGOMERY = _env_path("P2_MONTGOMERY", ROOT / "input" / "MontgomerySet")
WORK = ROOT / "pywork"                                           # everything we create
CKPT = WORK / "checkpoints"
RESULTS = WORK / "results"
MASKSTORE = WORK / "masks"                                       # saved explanation masks

for d in (WORK, CKPT, RESULTS, MASKSTORE):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- protocol
IMG_SIZE = 512            # working resolution for ROI construction (as in the paper)
NET_SIZE = 224            # network input (torchvision models; AlexNet also accepts 224)
DATA_SEED = 42            # fixes the train/held-out split; NEVER varied
TRAIN_SEEDS = [42, 43, 44]   # training-order/init seeds -- ordering stability (Concern 5)
SPLIT_HELDOUT = 0.20      # stratified

# faithful to the paper's recipe (MATLAB: sgdm, lr 1e-4, batch 10, 100 epochs,
# WeightLearnRateFactor 20 on the head, RandXReflection + +/-30px translation)
EPOCHS = 100
BATCH = 10
LR_BACKBONE = 1e-4
LR_HEAD = 20 * LR_BACKBONE
MOMENTUM = 0.9
TRANSLATE_FRAC = 30.0 / 512.0     # +/-30 px at 512, expressed as a fraction

ARCHS = ["alexnet", "vgg16", "vgg19", "resnet50"]
ARCHS_EXTRA = ["densenet121", "efficientnet_b0", "vit_b_16"]   # third family+ (Concern 5)

# ----------------------------------------------------------------------------- explainers
LIME_SAMPLES = 1000
LIME_KERNEL = 0.25
LIME_RUNS = 3
LIME_CONFIGS = [
    # key                  segmentation  n_seg  top-K
    ("LIME-sp-K30",        "slic",        50,   30),
    ("LIME-grid-K30",      "grid",        49,   30),
    ("LIME-fine-m100-K20", "slic",       100,   20),
]
GRADCAM_TAU = 0.5

# THE REVISION SWITCH (JMI Concern 1): superpixels over the whole frame
# ("full", reproduces the questioned setup) vs restricted to the lung/non-zero
# content ("lung", the reviewer's required recomputation). Runs produce BOTH.
SEG_SCOPES = ["full", "lung"]

# occlusion admissibility gate (JMI Concern 1b): report BOTH fills, always.
OCC_FILLS = ["mean", "zero"]
OCC_RAND_DRAWS = 5

# geometry nulls
NULL_DRAWS = 100

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from any hashable parts.

    Python's built-in hash() of strings is randomized per process
    (PYTHONHASHSEED), so it must never seed an experiment. This is CRC32 of
    the joined string form: stable across runs, machines and Python versions.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF
