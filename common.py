"""Shared code: data, models, losses, training and evaluation helpers."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

import config as C
from metrics_utils import cm_from_predictions, metrics_from_confusion


# --------------------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_name(model, loss, seed, split_dir):
    """Default setup (seed 42, folder 'splits') gives e.g. 'resnet50_weighted_ce'."""
    name = f"{model}_{loss}"
    if seed != C.SEED:
        name += f"_seed{seed}"
    split_name = Path(split_dir).name
    if split_name != "splits":
        name += f"_{split_name}"
    return name


# --------------------------------------------------------------------------- data
def _build_cache(raw):
    C.CACHE_DIR.mkdir(exist_ok=True)
    todo = [(stem, p) for stem, p in raw.items() if not (C.CACHE_DIR / f"{stem}.png").exists()]
    for stem, p in tqdm(todo, desc=f"Pre-resizing images to {C.IMAGE_SIZE}x{C.IMAGE_SIZE} (one-time)"):
        img = Image.open(p).convert("RGB").resize((C.IMAGE_SIZE, C.IMAGE_SIZE), Image.Resampling.BILINEAR)
        img.save(C.CACHE_DIR / f"{stem}.png")


def find_images(use_cache=True):
    """Return {image_id: path}. Uses the pre-resized cache (identical to resizing on the fly)."""
    raw = {p.stem: p for p in C.DATA_DIR.rglob("*.jpg")} if C.DATA_DIR.exists() else {}
    cache = {p.stem: p for p in C.CACHE_DIR.glob("*.png")} if C.CACHE_DIR.exists() else {}
    if not raw and not cache:
        raise FileNotFoundError(
            f"No HAM10000 images found in {C.DATA_DIR}. Download the dataset from Kaggle "
            "(kmader/skin-cancer-mnist-ham10000) and unzip it into the 'data' folder (see README).")
    if use_cache:
        if raw and len(cache) < len(raw):
            _build_cache(raw)
            cache = {p.stem: p for p in C.CACHE_DIR.glob("*.png")}
        if cache and (not raw or len(cache) >= len(raw)):
            return cache
    return raw


def load_splits(split_dir, image_paths):
    frames = []
    for name in ("train", "validation", "test"):
        df = pd.read_csv(Path(split_dir) / f"{name}_split.csv")
        df["dx"] = df["dx"].str.lower().str.strip()
        df["image_id"] = df["image_id"].astype(str)
        df["image_path"] = df["image_id"].map(lambda s: str(image_paths[s]) if s in image_paths else None)
        missing = int(df["image_path"].isna().sum())
        if missing:
            raise FileNotFoundError(f"{missing} images listed in {name}_split.csv were not found in {C.DATA_DIR}")
        frames.append(df)
    train_df, val_df, test_df = frames
    ids = [set(d["lesion_id"]) for d in frames]
    overlap = {"train_val": len(ids[0] & ids[1]), "train_test": len(ids[0] & ids[2]), "val_test": len(ids[1] & ids[2])}
    return train_df, val_df, test_df, overlap


IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {n: i for i, n in enumerate(C.CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[row["dx"]]


def make_loaders(train_df, val_df, test_df, micro_batch, num_workers, seed):
    train_tf, eval_tf = make_transforms()
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    drop_last = len(train_df) % micro_batch == 1  # BatchNorm cannot train on a batch of 1
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)
    train_loader = DataLoader(HAM10000Dataset(train_df, train_tf), batch_size=micro_batch, shuffle=True,
                              drop_last=drop_last, generator=g, **kw)
    val_loader = DataLoader(HAM10000Dataset(val_df, eval_tf), batch_size=micro_batch, shuffle=False, **kw)
    test_loader = DataLoader(HAM10000Dataset(test_df, eval_tf), batch_size=micro_batch, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- loss and handling class imbalance
def compute_class_weights(train_df):
    counts = train_df["dx"].value_counts().reindex(C.CLASS_NAMES, fill_value=0).values
    assert np.all(counts > 0), "A class is missing from the training split."
    return torch.tensor(len(train_df) / (C.NUM_CLASSES * counts), dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def make_criterion(loss_name, class_weights):
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=C.FOCAL_GAMMA)
    raise ValueError(f"Unknown loss: {loss_name}")


# --------------------------------------------------------------------------- models
class CustomCNN(nn.Module):
    """5 blocks of (Conv-BN-ReLU x2 + MaxPool), global average pooling, dropout, linear head."""

    def __init__(self, num_classes, widths=(32, 64, 128, 256, 256), dropout=0.4):
        super().__init__()
        blocks, in_ch = [], 3
        for w in widths:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            in_ch = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_model(name, pretrained=True):
    n = C.NUM_CLASSES
    if name == "custom_cnn":
        return CustomCNN(n)
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n)
        return m
    if name == "densenet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, n)
        return m
    if name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
        return m
    raise ValueError(f"Unknown model: {name}")


def get_target_layer(model, name):
    if name == "custom_cnn":
        return model.features[-1]
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet201":
        return model.features.denseblock4
    if name == "efficientnet_b3":
        return model.features[-1]
    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model, name):
    for p in model.parameters():
        p.requires_grad = False
    head = model.fc if name == "resnet50" else model.classifier
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- training / evaluation
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, accum_steps=1):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_items = 0.0, 0, 0
    n_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(tqdm(loader, leave=False, desc="train" if training else "val")):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            loss = criterion(logits.float(), labels)
            if training:
                scaler.scale(loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_items += bs
    return total_loss / total_items, total_correct / total_items


def fit_stage(model, train_loader, val_loader, criterion, device, epochs, lr, patience, stage,
              use_amp, accum_steps):
    """Train one stage; keep the weights of the epoch with the lowest validation loss."""
    if epochs <= 0:
        return model, []
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=C.LR_SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, accum_steps)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(va_loss)
        history.append(dict(stage=stage, epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=optimizer.param_groups[0]["lr"]))
        print(f"{stage} | epoch {epoch}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s", flush=True)
        if va_loss < best_loss:
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.", flush=True)
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, loader, device):
    """Full-precision predictions: returns y_true, y_pred, y_prob."""
    model.eval()
    ys, probs = [], []
    for images, labels in tqdm(loader, leave=False, desc="test"):
        logits = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(labels.numpy())
    y_prob = np.concatenate(probs)
    return np.concatenate(ys), y_prob.argmax(1), y_prob


def safe_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                   labels=np.arange(C.NUM_CLASSES)))
    except ValueError:
        return None


def summarize(y_true, y_pred, y_prob):
    cm = cm_from_predictions(y_true, y_pred)
    m = metrics_from_confusion(cm)
    return cm, m, safe_auc(y_true, y_prob)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
