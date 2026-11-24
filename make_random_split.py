"""Optional leakage ablation: build an IMAGE-level random split with the same sizes as the lesion-level one.

Run:  python make_random_split.py
Then: python train.py --model resnet50 --split-dir splits_random_image_level
Comparing its test scores with the lesion-level split shows how much leakage inflates results.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C

OUT = C.ROOT / "splits_random_image_level"


def main():
    parts = {n: pd.read_csv(C.SPLIT_DIR / f"{n}_split.csv") for n in ("train", "validation", "test")}
    full = pd.concat(parts.values(), ignore_index=True)
    n_val, n_test = len(parts["validation"]), len(parts["test"])
    dev, test = train_test_split(full, test_size=n_test, stratify=full["dx"], random_state=C.SEED)
    train, val = train_test_split(dev, test_size=n_val, stratify=dev["dx"], random_state=C.SEED)
    OUT.mkdir(exist_ok=True)
    for name, df in (("train", train), ("validation", val), ("test", test)):
        df.to_csv(OUT / f"{name}_split.csv", index=False)
    shared = len(set(train["lesion_id"]) & set(test["lesion_id"]))
    test_leaked = int(test["lesion_id"].isin(set(train["lesion_id"])).sum())
    print(f"Wrote {OUT}")
    print(f"sizes train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"lesions shared by train and test: {shared} ({test_leaked} of {len(test)} test images have a lesion also in train)")


if __name__ == "__main__":
    main()
