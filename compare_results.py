"""Build the final comparison (CNN from scratch vs transfer learning) from outputs/reports/*.

Run:  python compare_results.py
Writes: outputs/reports/results_table.csv / results_table.md / bootstrap_ci.csv and
        outputs/plots/comparison_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C
from metrics_utils import bootstrap_ci, metrics_from_confusion, predictions_from_confusion

REPORTS, PLOTS = C.OUTPUT_DIR / "reports", C.OUTPUT_DIR / "plots"
LABELS = {"custom_cnn": "Custom CNN (scratch)", "resnet50": "ResNet50 (transfer)",
          "densenet201": "DenseNet201 (transfer)", "efficientnet_b3": "EfficientNet-B3 (transfer)"}


def load_runs():
    runs = []
    for path in sorted(REPORTS.glob("*_summary.json")):
        s = json.load(open(path))
        s["run"] = path.name.replace("_summary.json", "")
        cm_path = REPORTS / f"{s['run']}_confusion_matrix.csv"
        s["cm"] = pd.read_csv(cm_path, index_col=0).values if cm_path.exists() else None
        runs.append(s)
    order = {m: i for i, m in enumerate(C.MODEL_ORDER)}
    runs.sort(key=lambda r: (r["split"], order.get(r["model"], 99), r["loss"], r["seed"]))
    return runs


def mean_sd(values, digits=3):
    v = [x for x in values if x is not None and not pd.isna(x)]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"


def results_table(runs):
    df = pd.DataFrame(runs)
    rows = []
    for (split, model, loss), g in df.groupby(["split", "model", "loss"], sort=False):
        mel = [r["per_class_recall"]["mel"] for r in g.to_dict("records")]
        rows.append({
            "Model": LABELS.get(model, model), "Loss": loss, "Split": split, "Seeds": len(g),
            "Accuracy": mean_sd(g["test_accuracy"]), "Macro-F1": mean_sd(g["test_macro_f1"]),
            "Weighted-F1": mean_sd(g["test_weighted_f1"]), "ROC-AUC": mean_sd(g["test_macro_ovr_roc_auc"]),
            "Mel recall": mean_sd(mel),
            "Params (M)": f"{g['trainable_params'].iloc[0] / 1e6:.1f}" if "trainable_params" in g and pd.notna(g["trainable_params"].iloc[0]) else "n/a",
            "Train min": mean_sd(g["train_minutes"], 0) if "train_minutes" in g else "n/a",
            "Source": g["source"].iloc[0],
        })
    return pd.DataFrame(rows)


def bootstrap_table(runs):
    rows = []
    for r in runs:
        npz = REPORTS / f"{r['run']}_predictions.npz"
        if npz.exists():
            d = np.load(npz); yt, yp, how = d["y_true"], d["y_pred"], "test predictions"
        elif r["cm"] is not None:
            yt, yp = predictions_from_confusion(r["cm"]); how = "rebuilt from confusion matrix"
        else:
            continue
        ci = bootstrap_ci(yt, yp)
        row = {"run": r["run"], "n_test": len(yt), "ci_source": how}
        for k, (lo, hi) in ci.items():
            row[f"{k}_95CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("No results found in outputs/reports. Run train.py or run_all.py first.")
    PLOTS.mkdir(parents=True, exist_ok=True)

    table = results_table(runs)
    table.to_csv(REPORTS / "results_table.csv", index=False)
    (REPORTS / "results_table.md").write_text(table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False))
    print(table.to_string(index=False))

    ci = bootstrap_table(runs)
    ci.to_csv(REPORTS / "bootstrap_ci.csv", index=False)
    print("\n95% bootstrap confidence intervals over test images:")
    print(ci.to_string(index=False))

    main_runs = [r for r in runs if r["split"] == "splits" and r["loss"] == "weighted_ce" and r["cm"] is not None]
    seen, first = set(), []
    for r in main_runs:
        if r["model"] not in seen:
            seen.add(r["model"]); first.append(r)
    if not first:
        return

    # 1) headline metrics
    names = [LABELS.get(r["model"], r["model"]) for r in first]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (key, title) in zip(ax, [("test_accuracy", "Accuracy"), ("test_macro_f1", "Macro-F1"), ("mel_recall", "Melanoma recall")]):
        vals = [r["per_class_recall"]["mel"] if key == "mel_recall" else r[key] for r in first]
        a.bar(range(len(vals)), vals, color=["#c0504d" if r["model_type"] == "scratch" else "#1f6f78" for r in first])
        a.set_xticks(range(len(vals))); a.set_xticklabels(names, rotation=20, ha="right")
        a.set_title(title); a.set_ylim(0, 1)
        for i, v in enumerate(vals):
            a.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(PLOTS / "comparison_headline_metrics.png", dpi=200); plt.close()

    # 2) per-class recall
    recall = pd.DataFrame({LABELS.get(r["model"], r["model"]): metrics_from_confusion(r["cm"])["recall"] for r in first},
                          index=C.CLASS_NAMES)
    ax = recall.plot(kind="bar", figsize=(11, 4.5), width=0.8)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1); ax.set_title("Per-class recall on the test set"); ax.legend(fontsize=8)
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(PLOTS / "comparison_per_class_recall.png", dpi=200); plt.close()

    # 3) normalised confusion matrices side by side
    fig, axes = plt.subplots(1, len(first), figsize=(5.2 * len(first), 4.6), squeeze=False)
    for a, r in zip(axes[0], first):
        norm = r["cm"] / r["cm"].sum(1, keepdims=True).clip(min=1)
        sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=a)
        a.set_title(LABELS.get(r["model"], r["model"]), fontsize=10); a.set_xlabel("Predicted"); a.set_ylabel("True")
    plt.tight_layout(); plt.savefig(PLOTS / "comparison_confusion_matrices.png", dpi=200); plt.close()
    print("\nSaved plots to", PLOTS)


def _has_tabulate():
    try:
        __import__("tabulate")
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
