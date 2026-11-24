"""Grad-CAM error analysis for a trained model.  

Needs outputs/checkpoints/<run>_best.pth (created by train.py). Produces 3 figures in outputs/gradcam/:
  <run>_correct_per_class.png   most confident correct prediction(s) of every class
  <run>_missed_melanoma.png     true melanoma predicted as nevus (the clinically worst error)
  <run>_false_melanoma.png      true nevus predicted as melanoma (the most common false alarm)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config as C
import common as K


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(C.MODEL_CONFIG))
    p.add_argument("--loss", default="weighted_ce", choices=list(C.LOSSES))
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--split-dir", default=str(C.SPLIT_DIR))
    p.add_argument("--per-class", type=int, default=2, help="correct examples per class")
    p.add_argument("--errors", type=int, default=8, help="examples per error figure")
    return p.parse_args()


def draw(df, indices, probs, preds, cam, eval_tf, device, title, path, ncols=4):
    if len(indices) == 0:
        print(f"  (no samples for: {title})")
        return
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(3.0 * ncols * 2, 3.2 * nrows), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for k, i in enumerate(indices):
        row = df.iloc[i]
        pil = Image.open(row["image_path"]).convert("RGB").resize((C.IMAGE_SIZE, C.IMAGE_SIZE))
        rgb = np.asarray(pil).astype(np.float32) / 255.0
        x = eval_tf(pil).unsqueeze(0).to(device)
        g = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(preds[i]))])[0]
        r, c = divmod(k, ncols)
        axes[r][2 * c].imshow(rgb)
        axes[r][2 * c].set_title(f"true: {row['dx']}", fontsize=9)
        axes[r][2 * c + 1].imshow(show_cam_on_image(rgb, g, use_rgb=True))
        axes[r][2 * c + 1].set_title(f"pred: {C.CLASS_NAMES[preds[i]]} ({probs[i, preds[i]]:.2f})", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("  saved", path)


def main():
    args = parse_args()
    name = K.run_name(args.model, args.loss, args.seed, args.split_dir)
    ckpt_path = C.OUTPUT_DIR / "checkpoints" / f"{name}_best.pth"
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first: python train.py --model {args.model}\n"
                         "(If you have the ResNet50 checkpoint from Colab, copy it to that exact path.)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = K.build_model(args.model, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    paths = K.find_images()
    _, _, test_df, _ = K.load_splits(args.split_dir, paths)
    _, eval_tf = K.make_transforms()
    _, _, loader = K.make_loaders(test_df, test_df, test_df, 8, 0, args.seed)
    y_true, y_pred, y_prob = K.predict(model, loader, device)
    cam = GradCAM(model=model, target_layers=[K.get_target_layer(model, args.model)])

    out = C.OUTPUT_DIR / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    conf = y_prob[np.arange(len(y_pred)), y_pred]
    mel, nv = C.CLASS_NAMES.index("mel"), C.CLASS_NAMES.index("nv")

    chosen = []
    for c in range(C.NUM_CLASSES):
        idx = np.where((y_true == c) & (y_pred == c))[0]
        chosen += list(idx[np.argsort(-conf[idx])][: args.per_class])
    draw(test_df, chosen, y_prob, y_pred, cam, eval_tf, device,
         f"{name}: most confident correct predictions", out / f"{name}_correct_per_class.png", ncols=4)

    for tag, t, p, title in (("missed_melanoma", mel, nv, "melanoma predicted as nevus"),
                             ("false_melanoma", nv, mel, "nevus predicted as melanoma")):
        idx = np.where((y_true == t) & (y_pred == p))[0]
        idx = idx[np.argsort(-conf[idx])][: args.errors]
        draw(test_df, list(idx), y_prob, y_pred, cam, eval_tf, device,
             f"{name}: {title} (most confident errors)", out / f"{name}_{tag}.png", ncols=4)


if __name__ == "__main__":
    main()
