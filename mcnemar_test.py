"""
Exact McNemar test between models, reading the project's *_predictions.npz files.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest

DEFAULTS = {
    "custom_cnn": "outputs/reports/custom_cnn_weighted_ce_predictions.npz",
    "densenet201": "outputs/reports/densenet201_weighted_ce_predictions.npz",
    "efficientnet_b3": "outputs/reports/efficientnet_b3_weighted_ce_predictions.npz",
}


def find_key(keys, words, exclude=()):
    for k in keys:
        kl = k.lower()
        if any(w in kl for w in words) and not any(e in kl for e in exclude):
            return k
    return None


def load(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print(f"{path}\n  arrays: " + ", ".join(f"{k}{d[k].shape}" for k in keys))
    kt = find_key(keys, ["true", "label", "target", "gt", "y_t"])
    kp = find_key(keys, ["pred"], exclude=["prob", "score", "logit"])
    if kt is None or kp is None:
        # fall back: pred as argmax of a probability/logit array
        kpr = find_key(keys, ["prob", "logit", "score"])
        if kt is None or kpr is None:
            sys.exit(f"Could not identify true/pred arrays in {path}. Paste the 'arrays:' line above.")
        y_pred = d[kpr].argmax(axis=1)
    else:
        y_pred = d[kp]
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
    y_true = d[kt]
    if y_true.ndim == 2:
        y_true = y_true.argmax(axis=1)
    return np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()


def main():
    args = [a for a in sys.argv[1:] if "=" in a]
    files = dict(a.split("=", 1) for a in args) if args else DEFAULTS
    runs = {name: load(path) for name, path in files.items()}

    rows = []
    for a, b in itertools.combinations(runs, 2):
        ta, pa = runs[a]
        tb, pb = runs[b]
        if len(ta) != len(tb) or not (ta == tb).all():
            sys.exit(f"{a} and {b} have different true labels/order, so they are not comparable.")
        ok_a, ok_b = pa == ta, pb == tb
        n10 = int((ok_a & ~ok_b).sum())
        n01 = int((~ok_a & ok_b).sum())
        n = n10 + n01
        p = 1.0 if n == 0 else binomtest(n10, n, 0.5).pvalue
        rows.append(dict(model_A=a, model_B=b, n_images=len(ta),
                         acc_A=round(ok_a.mean(), 4), acc_B=round(ok_b.mean(), 4),
                         only_A_correct=n10, only_B_correct=n01, p_value=p))

    res = pd.DataFrame(rows)
    res["p_bonferroni"] = (res["p_value"] * len(res)).clip(upper=1.0)
    res["significant_0.05"] = res["p_bonferroni"] < 0.05
    pd.set_option("display.width", 220)
    print("\n" + res.to_string(index=False))
    res.to_csv("outputs/reports/mcnemar_results.csv", index=False)
    print("\nSaved to outputs/reports/mcnemar_results.csv")


if __name__ == "__main__":
    main()