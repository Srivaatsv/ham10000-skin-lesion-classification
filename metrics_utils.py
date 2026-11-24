"""Torch-free metric helpers (work from a confusion matrix, so they also work for imported results)."""
import numpy as np
import pandas as pd
from config import CLASS_NAMES, NUM_CLASSES


def cm_from_predictions(y_true, y_pred, n=NUM_CLASSES):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
        "precision": precision, "recall": recall, "f1": f1, "support": support,
    }


def classification_report_df(cm):
    m = metrics_from_confusion(cm)
    df = pd.DataFrame({"precision": m["precision"], "recall": m["recall"],
                       "f1-score": m["f1"], "support": m["support"]}, index=CLASS_NAMES)
    df.loc["accuracy"] = [np.nan, np.nan, m["accuracy"], m["support"].sum()]
    df.loc["macro avg"] = [m["precision"].mean(), m["recall"].mean(), m["macro_f1"], m["support"].sum()]
    w = m["support"] / m["support"].sum()
    df.loc["weighted avg"] = [(m["precision"] * w).sum(), (m["recall"] * w).sum(),
                              m["weighted_f1"], m["support"].sum()]
    return df


def predictions_from_confusion(cm):
    """Expand a confusion matrix into (y_true, y_pred) arrays (sample order is lost, counts are exact)."""
    cm = np.asarray(cm, dtype=int)
    yt, yp = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            yt.extend([i] * cm[i, j]); yp.extend([j] * cm[i, j])
    return np.array(yt), np.array(yp)


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CIs over test images for the headline metrics."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n, mel = len(y_true), CLASS_NAMES.index("mel")
    draws = {"accuracy": [], "macro_f1": [], "weighted_f1": [], "mel_recall": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics_from_confusion(cm_from_predictions(y_true[idx], y_pred[idx]))
        draws["accuracy"].append(m["accuracy"]); draws["macro_f1"].append(m["macro_f1"])
        draws["weighted_f1"].append(m["weighted_f1"]); draws["mel_recall"].append(m["recall"][mel])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items()}
