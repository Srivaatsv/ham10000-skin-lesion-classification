"""Run the whole study.  Default:  python run_all.py

Trains every model that has no results yet (ResNet50 is already imported from the Colab run), then
builds the Grad-CAM figures and the comparison tables/plots. Safe to stop and restart: finished runs are skipped.

Options:
  --models custom_cnn densenet201 ...   subset of models
  --seeds 42 43 44                       repeat for mean +/- SD (recommended if time allows)
  --losses weighted_ce focal             also compare the focal loss
  --include-resnet50                     retrain ResNet50 locally under the shared config
  --ablation resnet50                    leakage ablation: train on a random image-level split
  --no-gradcam
"""
import argparse
import subprocess
import sys

import config as C
import common as K

PY = sys.executable


def sh(*cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([PY, *map(str, cmd)]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["custom_cnn", "densenet201", "efficientnet_b3"], choices=list(C.MODEL_CONFIG))
    ap.add_argument("--seeds", nargs="+", type=int, default=[C.SEED])
    ap.add_argument("--losses", nargs="+", default=["weighted_ce"], choices=list(C.LOSSES))
    ap.add_argument("--include-resnet50", action="store_true")
    ap.add_argument("--ablation", choices=list(C.MODEL_CONFIG), default=None)
    ap.add_argument("--no-gradcam", action="store_true")
    args = ap.parse_args()

    models = list(args.models)
    if args.include_resnet50 and "resnet50" not in models:
        models.insert(0, "resnet50")
    failed = []
    for seed in args.seeds:
        for loss in args.losses:
            for model in models:
                force = ["--force"] if (model == "resnet50" and args.include_resnet50 and seed == C.SEED
                                        and loss == "weighted_ce" and not (C.OUTPUT_DIR / "checkpoints" / "resnet50_weighted_ce_best.pth").exists()) else []
                if sh("train.py", "--model", model, "--loss", loss, "--seed", seed, *force):
                    failed.append(f"{model}/{loss}/seed{seed}")

    if args.ablation:
        sh("make_random_split.py")
        if sh("train.py", "--model", args.ablation, "--split-dir", C.ROOT / "splits_random_image_level"):
            failed.append(f"ablation/{args.ablation}")

    if not args.no_gradcam:
        for model in C.MODEL_ORDER:
            name = K.run_name(model, "weighted_ce", C.SEED, C.SPLIT_DIR)
            if (C.OUTPUT_DIR / "checkpoints" / f"{name}_best.pth").exists():
                sh("gradcam_analysis.py", "--model", model)
            else:
                print(f"(skipping Grad-CAM for {model}: no checkpoint)")
    sh("compare_results.py")
    print("\nFAILED RUNS:" if failed else "\nAll runs finished.", failed or "")


if __name__ == "__main__":
    main()
