"""Central configuration for the SignalScope ML pipeline (paths, label spaces, protocol, hyper-params)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
WEIGHTS_DIR = ROOT / "weights"
REPORT_DIR = ROOT / "report"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "signalscope_best.pt"
CALIBRATION_PATH = WEIGHTS_DIR / "calibration.json"
METRICS_PATH = REPORT_DIR / "metrics.json"

# --------------------------------------------------------------------------------------
# Label spaces
# --------------------------------------------------------------------------------------
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_FAKE: "ai_generated"}

# Every generator we know about (Tiny-GenImage label space + CIFAKE's SD1.4).
ALL_GENERATORS = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]

# Generator family taxonomy used for Bonus B (attribution) and for honest hedging on unseen generators.
GENERATOR_FAMILY = {
    "Real": "real",
    "ADM": "pixel-diffusion",
    "GLIDE": "pixel-diffusion",
    "VQDM": "vq-diffusion",
    "SD14": "latent-diffusion",
    "SD15": "latent-diffusion",
    "Wukong": "latent-diffusion",
    "Midjourney": "diffusion (proprietary)",
    "BigGAN": "gan",
}

# Unseen-generator protocol: these generators are NEVER used for training or calibration.
# Midjourney = the exact "Midjourney-class output" the brief cites as unseen; VQDM = a different diffusion
# formulation (vector-quantised).  Everything else is "seen".
HELD_OUT_GENERATORS = ["Midjourney", "VQDM"]
SEEN_GENERATORS = [g for g in ALL_GENERATORS if g not in HELD_OUT_GENERATORS]

# Attribution head label space = generators the model may legitimately learn (Real + seen fakes).
ATTRIBUTION_CLASSES = SEEN_GENERATORS  # ["Real","ADM","BigGAN","GLIDE","SD14","SD15","Wukong"]
ATTR_INDEX = {g: i for i, g in enumerate(ATTRIBUTION_CLASSES)}
ATTR_IGNORE = -1

# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------
PREP_SHORT_SIDE = 256          # uniform storage resolution for high-res sources (removes resolution shortcut)
PREP_JPEG_QUALITY = 95         # uniform re-encode for BOTH classes (removes file-format shortcut)
CIFAKE_TRAIN_PER_CLASS = 20000  # sub-sample CIFAKE so 32px data doesn't drown the high-res data
CIFAKE_VAL_PER_CLASS = 1000
VAL_FRACTION = 0.10             # carved from train (model selection + temperature scaling)
GENIMAGE_TEST_FRACTION = 0.20   # share of seen-generator GenImage images held out as local test (protocol=seen)
REAL_TO_TEST_FRACTION = 0.10    # extra share of GenImage REAL images routed to test to balance test reals
SEED = 1337

# --------------------------------------------------------------------------------------
# Model / training defaults
# --------------------------------------------------------------------------------------
BACKBONE = "efficientnet_b0"
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 64
DEFAULT_LR = 3e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
ATTR_LOSS_WEIGHT = 0.3

# Operating point: chosen on the validation split to keep the false-positive rate (real flagged as AI) low.
TARGET_FPR = 0.05
# Calibrated-probability bands used for responsible presentation.
UNCERTAIN_BAND = 0.15  # |p - threshold| < band  ->  "uncertain"
