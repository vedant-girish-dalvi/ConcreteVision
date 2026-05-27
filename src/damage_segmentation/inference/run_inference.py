"""
dacl10k Inference Script — testchallenge / testdev
====================================================
Runs your trained segmentation model over the dacl10k test images and saves
predictions in the exact .npy format expected by prepare_submission.py.

Output per image:
    <pred_dir>/<image_stem>.npy   shape (H, W, 19)  dtype float32
    Values are per-class sigmoid probabilities in [0, 1].
    prepare_submission.py will threshold these at 0.5 to produce the final masks.

Usage:
    python run_inference.py \
        --checkpoint  /path/to/your/model.pth \
        --data_dir    /path/to/dacl10k \
        --pred_dir    /path/to/output/predictions \
        --split       testchallenge \
        --img_size    512 \
        --batch_size  8 \
        --device      cuda

Requirements:
    pip install torch torchvision pillow numpy tqdm albumentations
    (adjust imports in the MODEL DEFINITION section for your architecture)

===========================================================================
  *** READ: THREE PLACES YOU MUST EDIT FOR YOUR SPECIFIC MODEL ***
  1. SECTION A — imports:    add your model class / library
  2. SECTION B — load model: instantiate and load your checkpoint
  3. SECTION C — preprocess: match the normalisation you used during training
===========================================================================
"""

import os
import argparse
import numpy as np
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from model import SegmentationModel

# -----------------------------------------------------------------
# SECTION A — MODEL IMPORTS
# -----------------------------------------------------------------
# Replace or extend this block with your own model imports.
#
# Examples:
#   from segmentation_models_pytorch import Unet, DeepLabV3Plus, FPN
#   from transformers import SegformerForSemanticSegmentation
#   from my_model import MyCustomModel
#
# The only contract: your model must accept a (B, 3, H, W) float32 tensor
# and return logits of shape (B, 19, H, W)  OR  (B, 19, H_out, W_out).
# -----------------------
# Read config
# -----------------------
with open("config_training_rtx4500.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# shorthand
TRAIN_CFG = cfg.get("training", {})
MODEL_CFG = cfg.get("model", {})
OPT_CFG = cfg.get("optimizer", {})
SCHED_CFG = cfg.get("scheduler", {})
AMP_CFG = cfg.get("amp", {})
CKPT_CFG = cfg.get("checkpoints", {})
EARLY_CFG = cfg.get("early_stopping", {})

# -----------------------
# Hyperparams
# -----------------------
ARCHITECTURE = MODEL_CFG.get("architecture", "DeepLabV3Plus")
ENCODER = MODEL_CFG.get("encoder", "mit_b5")
WEIGHTS = MODEL_CFG.get("weights", "imagenet")
NUM_CLASSES = MODEL_CFG.get("num_classes", 19)

# -----------------------------------------------------------------
# dacl10k class list  (19 classes, index = channel)
# -----------------------------------------------------------------
CLASS_NAMES = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust",
    "Rockpocket", "Hollowareas", "Cavity", "Spalling", "Graffiti",
    "Weathering", "Restformwork", "ExposedRebars", "Bearing",
    "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
]
NUM_CLASSES = len(CLASS_NAMES)


# =================================================================
# SECTION B — MODEL LOADING
# =================================================================
# Edit this function to match your architecture and checkpoint format.
# Must return a model already in eval() mode on the correct device.
# =================================================================

def load_model(checkpoint_path: str, device: torch.device, img_size: int) -> torch.nn.Module:
    """
    Load your trained model from a checkpoint file.

    ---- EDIT THIS FUNCTION ----

    The example below covers two common cases:
      1. segmentation_models_pytorch (smp) — e.g. Unet, FPN, DeepLabV3+
      2. A plain state_dict saved with torch.save(model.state_dict(), ...)

    Replace with whatever instantiation your training script used.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ------------------------------------------------------------------
    # Example 1: segmentation_models_pytorch model
    # Uncomment and adapt encoder_name / architecture to match yours.
    # ------------------------------------------------------------------
    # model = smp.Unet(
    #     encoder_name="resnet50",        # e.g. "efficientnet-b4", "mit_b5"
    #     encoder_weights=None,           # weights come from checkpoint
    #     in_channels=3,
    #     classes=NUM_CLASSES,
    #     activation=None,                # we apply sigmoid manually
    # )

    # ------------------------------------------------------------------
    # Example 2: HuggingFace SegFormer
    # ------------------------------------------------------------------
    # from transformers import SegformerForSemanticSegmentation
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     "nvidia/mit-b5",
    #     num_labels=NUM_CLASSES,
    #     ignore_mismatched_sizes=True,
    # )

    # ------------------------------------------------------------------
    # Example 3: Your own custom model class
    # ------------------------------------------------------------------
    # from my_model import MySegmentationModel
    # model = MySegmentationModel(num_classes=NUM_CLASSES)

    # ------------------------------------------------------------------
    # Load weights (handles both full checkpoint dicts and bare state_dicts)
    # ------------------------------------------------------------------
    model = SegmentationModel(arch=ARCHITECTURE, encoder=ENCODER, weights=WEIGHTS, num_classes=NUM_CLASSES)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Assume the file is a bare state_dict
        state_dict = checkpoint

    # Strip "module." prefix from DataParallel / DistributedDataParallel checkpoints
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =================================================================
# SECTION C — PREPROCESSING
# =================================================================
# Match EXACTLY the normalisation you used during training.
# The defaults below (ImageNet mean/std, resize to square) are the
# most common choice for dacl10k work. Adjust if you did something
# different (e.g. different resize, extra augmentations at test time).
# =================================================================

# ImageNet statistics — change if you normalised differently
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_image(pil_img: Image.Image, img_size: int) -> torch.Tensor:
    """
    Convert a PIL image to a (1, 3, img_size, img_size) float32 tensor.

    Edit this if you used a different resize strategy or normalisation.
    For example, if you used albumentations during training, replicate
    the same pipeline here (without random augmentations).
    """
    # Resize — BILINEAR matches most training pipelines; use BICUBIC for SegFormer
    img = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0                  # [0, 1]

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std                                      # normalise

    tensor = torch.from_numpy(arr).permute(2, 0, 1)               # (3, H, W)
    return tensor.unsqueeze(0)                                     # (1, 3, H, W)


# -----------------------------------------------------------------
# Dataset — no annotations needed for test images
# -----------------------------------------------------------------

class DaclTestDataset(Dataset):
    """Loads raw test images from images/<split>/ with no ground truth."""

    def __init__(self, data_dir: str, split: str, img_size: int):
        self.img_dir  = Path(data_dir) / "images" / split
        self.img_size = img_size
        self.paths    = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {self.img_dir}")
        print(f"Found {len(self.paths)} images in '{split}' split.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        img   = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size          # PIL: (W, H)
        tensor = preprocess_image(img, self.img_size).squeeze(0)  # (3, H, W)
        return {
            "tensor": tensor,
            "stem":   path.stem,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }


# -----------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------

@torch.no_grad()
def check_disk_space(pred_dir: Path, n_images: int, orig_h: int, orig_w: int, save_format: str):
    """
    Estimate required disk space and warn if free space is insufficient.
    Uses a sample image resolution as a rough estimate.
    """
    import shutil
    bytes_per_pixel = 1 if save_format == "uint8" else 4   # uint8=1B, float32=4B
    est_bytes = n_images * orig_h * orig_w * 19 * bytes_per_pixel
    free_bytes = shutil.disk_usage(pred_dir).free
    est_gb  = est_bytes  / 1e9
    free_gb = free_bytes / 1e9
    print(f"  Estimated output size : {est_gb:.2f} GB  ({save_format})")
    print(f"  Free disk space       : {free_gb:.2f} GB")
    if est_bytes > free_bytes * 0.9:
        raise RuntimeError(
            f"\n  ✗ Insufficient disk space!\n"
            f"    Need ~{est_gb:.1f} GB, only {free_gb:.1f} GB available.\n"
            f"    Solutions:\n"
            f"      1. Use --save_format uint8  (4× smaller, already the default)\n"
            f"      2. Free up space on your drive\n"
            f"      3. Point --pred_dir to a drive with more space"
        )


def save_prediction(out_path: Path, prob_hwc: np.ndarray, save_format: str):
    """
    Save a (H, W, 19) probability array to disk.

    save_format options:
      'uint8'   — stores probabilities as 0-255 uint8  (÷255 to recover).
                  4× smaller than float32. Default and recommended.
                  prepare_submission.py loads and thresholds at 127 (≈0.5).
      'float32' — stores raw float32 probabilities. Largest but lossless.
    """
    if save_format == "uint8":
        arr = (prob_hwc * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = prob_hwc.astype(np.float32)
    np.save(out_path, arr)


def run_inference(
    model:       torch.nn.Module,
    dataset:     DaclTestDataset,
    pred_dir:    Path,
    batch_size:  int,
    device:      torch.device,
    img_size:    int,
    tta:         bool = False,
    save_format: str  = "uint8",
    resume:      bool = True,
):
    """
    Run model inference and save per-image .npy prediction files.

    Parameters
    ----------
    tta : bool
        If True, applies horizontal-flip test-time augmentation (TTA)
        and averages the two predictions. Typically adds ~0.5-1 mIoU
        at the cost of 2× inference time.
    save_format : str
        'uint8'   — probabilities stored as 0-255 (recommended, 4× smaller).
        'float32' — raw float32 probabilities (lossless but large).
    resume : bool
        If True, skip images whose .npy file already exists in pred_dir.
        Useful for continuing an interrupted run without reprocessing.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-completed predictions when resuming
    if resume:
        pending = [
            item for item in dataset
            if not (pred_dir / (item["stem"] + ".npy")).exists()
        ]
        skipped = len(dataset) - len(pending)
        if skipped:
            print(f"  Resuming: skipping {skipped} already-completed predictions.")
        items_to_run = pending
    else:
        items_to_run = list(dataset)

    if not items_to_run:
        print("  All predictions already exist. Nothing to do.")
        return 0

    # Disk space check using first image's resolution as estimate
    sample = items_to_run[0]
    check_disk_space(pred_dir, len(items_to_run), sample["orig_h"], sample["orig_w"], save_format)

    loader = DataLoader(
        items_to_run,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,        # 0 = safer on Windows; increase on Linux if needed
        pin_memory=(device.type == "cuda"),
    )

    processed, failed = 0, []
    for batch in tqdm(loader, desc="Running inference"):
        tensors = batch["tensor"].to(device)      # (B, 3, H, W)
        stems   = batch["stem"]
        orig_hs = batch["orig_h"].tolist()
        orig_ws = batch["orig_w"].tolist()

        # ---- forward pass ----
        logits = model_forward(model, tensors, img_size)  # (B, 19, H, W)

        if tta:
            logits_flip = model_forward(model, tensors.flip(-1), img_size)
            logits = (logits + logits_flip.flip(-1)) * 0.5

        probs = torch.sigmoid(logits).cpu().float()        # (B, 19, H, W)

        for i, stem in enumerate(stems):
            orig_h = orig_hs[i]
            orig_w = orig_ws[i]

            # Resize probabilities back to original image resolution
            prob_i = probs[i].unsqueeze(0)                 # (1, 19, H, W)
            prob_i = F.interpolate(
                prob_i, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False
            ).squeeze(0)                                   # (19, orig_h, orig_w)

            # (19, H, W) -> (H, W, 19)
            out = prob_i.permute(1, 2, 0).detach().cpu().numpy()

            out_path = pred_dir / (stem + ".npy")
            try:
                save_prediction(out_path, out, save_format)
                processed += 1
            except OSError as e:
                failed.append(stem)
                tqdm.write(f"  [ERROR] Failed to save {stem}: {e}")
                # Check if disk is now full and abort early
                import shutil
                if shutil.disk_usage(pred_dir).free < 100 * 1024 * 1024:  # < 100 MB
                    print("\n  ✗ Disk is full. Stopping early.")
                    print(f"  ✓ Successfully saved {processed} predictions before stopping.")
                    print(f"  Re-run with --resume to continue from where you left off.")
                    break

    print(f"\n✓ Saved {processed} prediction files to: {pred_dir}")
    if failed:
        print(f"⚠ Failed to save {len(failed)} files: {failed[:5]}"
              + (" ..." if len(failed) > 5 else ""))
        print("  Re-run the script with --resume to retry failed files.")
    return processed


def model_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    """
    Unified forward pass that handles different output shapes.

    - Standard models (smp, custom): output shape (B, 19, H, W)  → use as-is
    - SegFormer (HuggingFace):        output shape (B, 19, H/4, W/4) → upsample

    Edit this function if your model has a non-standard output interface
    (e.g. returns a dict, a tuple, or needs a different upsampling strategy).
    """
    output = model(x)

    # HuggingFace SegFormer returns a ModelOutput object
    if hasattr(output, "logits"):
        logits = output.logits   # (B, 19, H/4, W/4)
    elif isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        logits = output           # (B, 19, H, W)

    # Upsample to input resolution if the model downsampled (e.g. SegFormer)
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(
            logits, size=x.shape[-2:],
            mode="bilinear", align_corners=False
        )

    return logits                 # (B, 19, img_size, img_size)


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dacl10k segmentation inference and save .npy predictions."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to your trained model checkpoint (.pth / .pt)."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root directory of the dacl10k dataset (contains images/<split>/)."
    )
    parser.add_argument(
        "--pred_dir", required=True,
        help="Directory where prediction .npy files will be saved."
    )
    parser.add_argument(
        "--split", default="testchallenge",
        choices=["testdev", "testchallenge", "validation", "train"],
        help="Dataset split to run inference on (default: testchallenge)."
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
        help="Square size images are resized to before being fed to the model "
             "(default: 512). Must match your training configuration."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size (default: 8). Reduce if you run out of VRAM."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device: 'cuda', 'cuda:1', 'mps', or 'cpu' (auto-detected)."
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable horizontal-flip test-time augmentation (TTA). "
             "Slightly improves mIoU at the cost of 2x inference time."
    )
    parser.add_argument(
        "--save_format", default="uint8", choices=["uint8", "float32"],
        help=(
            "Format for saved .npy prediction files (default: uint8). "
            "uint8 stores probabilities as 0-255 integers -- 4x smaller than float32 "
            "and fully compatible with prepare_submission.py. "
            "Use float32 only if you need lossless probabilities downstream."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip images whose .npy already exists in pred_dir (default: on). "
             "Lets you continue an interrupted run without reprocessing."
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Reprocess all images even if predictions already exist."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    pred_dir = Path(args.pred_dir)

    print("=" * 60)
    print("  dacl10k Inference")
    print("=" * 60)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Split       : {args.split}")
    print(f"  Image size  : {args.img_size}x{args.img_size}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Device      : {device}")
    print(f"  TTA         : {args.tta}")
    print(f"  Save format : {args.save_format}")
    print(f"  Resume      : {args.resume}")
    print(f"  Output dir  : {pred_dir}")
    print()

    # 1. Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device, args.img_size)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {param_count:.1f}M")

    # 2. Build dataset
    dataset = DaclTestDataset(args.data_dir, args.split, args.img_size)

    # 3. Run inference
    n = run_inference(
        model, dataset, pred_dir,
        batch_size=args.batch_size,
        device=device,
        img_size=args.img_size,
        tta=args.tta,
        save_format=args.save_format,
        resume=args.resume,
    )

    # 4. Print next step
    print("\nNext step -- package for CodaLab submission:")
    print(f"  python prepare_submission.py \\")
    print(f"      --pred_dir  {pred_dir} \\")
    print(f"      --data_dir  {args.data_dir} \\")
    print(f"      --split     {args.split} \\")
    print(f"      --out_zip   submission_{args.split}.zip")


if __name__ == "__main__":
    main()