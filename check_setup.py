"""Run this FIRST:  python check_setup.py        (add --memory-test to test GPU memory for every model)"""
import argparse
import sys


def memory_test(torch, K, C, name, micro_batch):
    """One mixed-precision training step; returns peak GB, or None if the GPU ran out of memory."""
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = K.build_model(name, pretrained=False).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.randn(micro_batch, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, device=dev)
        y = torch.zeros(micro_batch, dtype=torch.long, device=dev)
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        return torch.cuda.max_memory_allocated() / 1024**3
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-test", action="store_true")
    args = ap.parse_args()
    problems = []
    print("Python", sys.version.split()[0])
    try:
        import torch, torchvision
        print("torch", torch.__version__, "| torchvision", torchvision.__version__)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"GPU: {p.name}, {p.total_memory / 1024**3:.1f} GB")
        else:
            problems.append("CUDA GPU not available to PyTorch (training would be extremely slow on CPU). "
                            "Install the CUDA build of torch - see README step 2.")
    except ImportError as e:
        raise SystemExit(f"PyTorch is not installed: {e}. See README step 2.")
    for mod in ("numpy", "pandas", "sklearn", "matplotlib", "seaborn", "PIL", "tqdm", "pytorch_grad_cam"):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"missing package '{mod}': run  pip install -r requirements.txt")

    import config as C
    import common as K
    try:
        paths = K.find_images()
        print(f"Images found: {len(paths)} (expected 10015)")
        if len(paths) < 10015:
            problems.append("Fewer than 10015 images found - make sure BOTH image folders (part_1 and part_2) are in data/")
        tr, va, te, overlap = K.load_splits(C.SPLIT_DIR, paths)
        print("Split sizes train/val/test:", len(tr), len(va), len(te), "(expected 6376/1613/2026)")
        print("Lesion overlap between splits:", overlap, "(must be all 0)")
        if (len(tr), len(va), len(te)) != (6376, 1613, 2026) or sum(overlap.values()):
            problems.append("split files are not the expected fixed split - do not edit the files in splits/")
    except Exception as e:  # noqa: BLE001
        problems.append(str(e))

    if args.memory_test and torch.cuda.is_available():
        for name, cfg in C.MODEL_CONFIG.items():
            peak = memory_test(torch, K, C, name, cfg["micro_batch"])
            if peak is None:
                problems.append(f"{name} runs out of GPU memory at micro-batch {cfg['micro_batch']} - lower it in config.py")
            else:
                print(f"  {name:16s} micro-batch {cfg['micro_batch']:2d}: peak GPU memory {peak:.2f} GB  OK")

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(" -", p)
    else:
        print("READY. Next step: python train.py --model custom_cnn --smoke-test   (about 1-2 minutes)")


if __name__ == "__main__":
    main()
