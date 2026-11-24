"""Single place for every setting. Change things here, not inside the other scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # put the downloaded HAM10000 images here (any sub-folders are fine)
SPLIT_DIR = ROOT / "splits"         # fixed lesion-level split used for ALL models
OUTPUT_DIR = ROOT / "outputs"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
CACHE_DIR = ROOT / f"data_cache_{IMAGE_SIZE}"  # auto-created: images pre-resized once (much faster training)
SEED = 42

# Every model is trained with the SAME effective batch size. Models that do not fit the
# 4 GB GPU with 16 images use a smaller micro-batch plus gradient accumulation.
EFFECTIVE_BATCH_SIZE = 16
NUM_WORKERS = 0           # set to 0 if you run out of RAM (8 GB machines) or see Windows worker errors
WEIGHT_DECAY = 1e-4
LR_SCHEDULER_PATIENCE = 2  # ReduceLROnPlateau(factor=0.5) on validation loss
FOCAL_GAMMA = 2.0
LOSSES = ("weighted_ce", "focal")

# Transfer-learning models: stage 1 trains only the new classifier head, stage 2 fine-tunes everything.
# Custom CNN: trained from scratch (stage 2 only), with a longer budget.
# NOTE: head_lr / finetune_lr / patience for the transfer models were not recoverable from the
# original Colab notebook. Replace them with your Colab values if they differ (see README).
MODEL_CONFIG = {
    "custom_cnn":      dict(kind="scratch",  pretrained=False, head_epochs=0, finetune_epochs=80,
                            head_lr=None, finetune_lr=1e-3, patience=10, micro_batch=16),
    "resnet50":        dict(kind="transfer", pretrained=True,  head_epochs=5, finetune_epochs=10,
                            head_lr=1e-3, finetune_lr=1e-4, patience=5, micro_batch=16),
    "densenet201":     dict(kind="transfer", pretrained=True,  head_epochs=5, finetune_epochs=10,
                            head_lr=1e-3, finetune_lr=1e-4, patience=5, micro_batch=8),
    "efficientnet_b3": dict(kind="transfer", pretrained=True,  head_epochs=5, finetune_epochs=10,
                            head_lr=1e-3, finetune_lr=1e-4, patience=5, micro_batch=8),
}
MODEL_ORDER = ["custom_cnn", "resnet50", "densenet201", "efficientnet_b3"]
