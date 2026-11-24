"""Train and evaluate ONE model.  Example:  python train.py --model efficientnet_b3

Outputs (in outputs/): reports/<run>_summary.json, *_classification_report.csv, *_confusion_matrix.csv,
*_training_history.csv, *_predictions.npz; plots/<run>_training_curves.png, *_confusion_matrix.png;
checkpoints/<run>_best.pth
"""
import argparse
import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import config as C
import common as K
from metrics_utils import classification_report_df


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate one model on HAM10000.")
    p.add_argument("--model", required=True, choices=list(C.MODEL_CONFIG))
    p.add_argument("--loss", default="weighted_ce", choices=list(C.LOSSES))
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--split-dir", default=str(C.SPLIT_DIR), help="folder with train/validation/test_split.csv")
    p.add_argument("--micro-batch", type=int, default=None, help="override the per-step batch size")
    p.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--no-cache", action="store_true", help="read the original JPEGs instead of the 224px cache")
    p.add_argument("--force", action="store_true", help="retrain even if results already exist")
    p.add_argument("--smoke-test", action="store_true", help="tiny 1-epoch run to verify the setup (not saved to reports)")
    return p.parse_args()


def train_once(args, micro_batch):
    cfg = C.MODEL_CONFIG[args.model]
    name = K.run_name(args.model, args.loss, args.seed, args.split_dir)
    out = C.OUTPUT_DIR / ("smoke" if args.smoke_test else "")
    for sub in ("reports", "plots", "checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    K.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = device.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    accum = max(1, C.EFFECTIVE_BATCH_SIZE // micro_batch)
    print(f"\n=== {name} | device: {gpu} | micro-batch {micro_batch} x accum {accum} | amp: {use_amp} ===", flush=True)

    image_paths = K.find_images(use_cache=not args.no_cache)
    train_df, val_df, test_df, overlap = K.load_splits(args.split_dir, image_paths)
    print("Images train/val/test:", len(train_df), len(val_df), len(test_df), "| lesion overlap:", overlap)
    if Path(args.split_dir).name == "splits":
        assert sum(overlap.values()) == 0, "The shipped split must have zero lesion overlap."

    class_weights = K.compute_class_weights(train_df)  # from the FULL training split
    if args.smoke_test:
        train_df = train_df.sample(96, random_state=0)
        val_df, test_df = val_df.sample(48, random_state=0), test_df.sample(48, random_state=0)
    train_loader, val_loader, test_loader = K.make_loaders(
        train_df, val_df, test_df, micro_batch, 0 if args.smoke_test else args.num_workers, args.seed)

    criterion = K.make_criterion(args.loss, class_weights.to(device))
    model = K.build_model(args.model, pretrained=cfg["pretrained"]).to(device)
    n_params = K.count_params(model)
    head_epochs = 1 if args.smoke_test and cfg["head_epochs"] > 0 else cfg["head_epochs"]
    ft_epochs = 1 if args.smoke_test else cfg["finetune_epochs"]

    t0 = time.time()
    history = []
    if head_epochs > 0:
        K.freeze_backbone(model, args.model)
        model, h = K.fit_stage(model, train_loader, val_loader, criterion, device, head_epochs,
                               cfg["head_lr"], cfg["patience"], "head", use_amp, accum)
        history += h
        K.unfreeze_all(model)
    model, h = K.fit_stage(model, train_loader, val_loader, criterion, device, ft_epochs,
                           cfg["finetune_lr"], cfg["patience"], "fine_tune", use_amp, accum)
    history += h
    train_minutes = (time.time() - t0) / 60

    hist = pd.DataFrame(history)
    hist.to_csv(out / "reports" / f"{name}_training_history.csv", index=False)
    torch.save({"model_name": args.model, "loss_name": args.loss, "class_names": C.CLASS_NAMES,
                "model_state_dict": model.state_dict(), "image_size": C.IMAGE_SIZE},
               out / "checkpoints" / f"{name}_best.pth")

    # ---- evaluation on the untouched test split
    y_true, y_pred, y_prob = K.predict(model, test_loader, device)
    cm, m, auc = K.summarize(y_true, y_pred, y_prob)
    np.savez_compressed(out / "reports" / f"{name}_predictions.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    pd.DataFrame(cm, index=C.CLASS_NAMES, columns=C.CLASS_NAMES).to_csv(out / "reports" / f"{name}_confusion_matrix.csv")
    report = classification_report_df(cm)
    report.to_csv(out / "reports" / f"{name}_classification_report.csv")
    print("\n" + report.round(3).to_string())
    print(f"\nTest accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"weighted-F1 {m['weighted_f1']:.4f} | ROC-AUC {auc if auc is None else round(auc, 4)}")

    summary = {
        "source": "local_run", "model": args.model, "model_type": cfg["kind"], "loss": args.loss,
        "seed": args.seed, "split": Path(args.split_dir).name, "image_size": C.IMAGE_SIZE,
        "effective_batch_size": C.EFFECTIVE_BATCH_SIZE, "micro_batch": micro_batch, "accum_steps": accum,
        "mixed_precision": use_amp, "head_epochs_requested": head_epochs, "finetune_epochs_requested": ft_epochs,
        "epochs_run": int(len(hist)), "trainable_params": int(n_params), "train_minutes": round(train_minutes, 1),
        "device": gpu,
        "test_accuracy": float(m["accuracy"]), "test_macro_f1": float(m["macro_f1"]),
        "test_weighted_f1": float(m["weighted_f1"]), "test_macro_ovr_roc_auc": auc,
        "mel_recall": float(m["recall"][C.CLASS_NAMES.index("mel")]),
        "per_class_recall": {c: float(r) for c, r in zip(C.CLASS_NAMES, m["recall"])},
        "per_class_precision": {c: float(r) for c, r in zip(C.CLASS_NAMES, m["precision"])},
        "per_class_f1": {c: float(r) for c, r in zip(C.CLASS_NAMES, m["f1"])},
        "train_images": int(len(train_df)), "validation_images": int(len(val_df)), "test_images": int(len(test_df)),
    }
    K.save_json(summary, out / "reports" / f"{name}_summary.json")

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, col, title in zip(ax, ("loss", "acc"), ("Loss", "Accuracy")):
        a.plot(hist[f"train_{col}"].values, label="Train")
        a.plot(hist[f"val_{col}"].values, label="Validation")
        a.set_title(f"{args.model} - {title}"); a.set_xlabel("Epoch"); a.legend()
    plt.tight_layout()
    plt.savefig(out / "plots" / f"{name}_training_curves.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=ax[0])
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=ax[1])
    ax[0].set_title(f"{args.model} - confusion matrix (counts)")
    ax[1].set_title("Row-normalised (= per-class recall on the diagonal)")
    for a in ax:
        a.set_xlabel("Predicted label"); a.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(out / "plots" / f"{name}_confusion_matrix.png", dpi=200)
    plt.close()
    print(f"Saved results for {name} under {out}")


def main():
    args = parse_args()
    name = K.run_name(args.model, args.loss, args.seed, args.split_dir)
    if not args.force and not args.smoke_test and (C.OUTPUT_DIR / "reports" / f"{name}_summary.json").exists():
        print(f"Results for {name} already exist - skipping (use --force to retrain).")
        return
    micro = args.micro_batch or C.MODEL_CONFIG[args.model]["micro_batch"]
    while True:
        out_of_memory = False
        try:
            train_once(args, micro)
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True
        if not out_of_memory:
            break
        gc.collect()
        torch.cuda.empty_cache()
        if micro <= 2:
            raise SystemExit("Out of GPU memory even with a micro-batch of 2. Close other GPU programs "
                             "(browsers, games) or reduce IMAGE_SIZE in config.py.")
        micro //= 2
        print(f"\nGPU ran out of memory - retrying with micro-batch {micro} "
              f"(gradient accumulation keeps the effective batch at {C.EFFECTIVE_BATCH_SIZE}).", flush=True)


if __name__ == "__main__":
    main()
