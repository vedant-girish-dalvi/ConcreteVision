"""
dacl10k mIoU Evaluation Script
================================
Computes per-class IoU and mean IoU (mIoU) for your model's predictions
against the dacl10k validation set ground truth.

Usage:
    python evaluate_miou.py \
        --data_dir /path/to/dacl10k \
        --pred_dir /path/to/your/predictions \
        --split    validation

If your dataset is not in the standard layout, use the override flags:
    python evaluate_miou.py \
        --pred_dir   C:/data/predictions \
        --ann_dir    C:/data/dacl10k/annotations/validation \
        --images_dir C:/data/dacl10k/images/validation \
        --split      validation

Requirements:
    pip install numpy pillow tqdm
    pip install git+https://github.com/phiyodr/dacl10k-toolkit
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# -----------------------------------------------------------------
# dacl10k class definitions (19 classes, same order as toolkit)
# -----------------------------------------------------------------
CLASS_NAMES = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust",
    "Rockpocket", "Hollowareas", "Cavity", "Spalling", "Graffiti",
    "Weathering", "Restformwork", "ExposedRebars", "Bearing",
    "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
]
NUM_CLASSES = len(CLASS_NAMES)  # 19


# -----------------------------------------------------------------
# Mask loading helpers
# -----------------------------------------------------------------

def load_gt_mask_from_annotation(annotation_path: Path, image_shape: tuple) -> np.ndarray:
    from dacl10k.utils import labelme2mask
    mask = labelme2mask(str(annotation_path))

    # Debug: print shape on first call to understand the output format
    # Remove these print lines once working
    # print(f"\n  [DEBUG] labelme2mask output shape: {mask.shape}, dtype: {mask.dtype}")

    # Handle different possible output shapes from different toolkit versions:
    # (19, H, W) -> transpose to (H, W, 19)
    if mask.ndim == 3 and mask.shape[0] == NUM_CLASSES:
        return mask.astype(bool).transpose(1, 2, 0)
    # (H, W, 19) -> already correct
    elif mask.ndim == 3 and mask.shape[2] == NUM_CLASSES:
        return mask.astype(bool)
    # (H*W, 19) or (H, 19) -> unexpected, print and raise
    else:
        raise ValueError(
            f"Unexpected mask shape from labelme2mask: {mask.shape}. "
            f"Expected (19, H, W) or (H, W, 19)."
        )


def load_pred_mask(pred_path: Path) -> np.ndarray:
    """
    Load a prediction mask and return a (H, W, 19) boolean array.

    Supports:
      .npy -- shape (H, W, 19) or (19, H, W)
              dtype uint8  (0-255, from run_inference.py --save_format uint8)
              dtype float32 (0.0-1.0, from run_inference.py --save_format float32)
              dtype bool
      .png -- single-channel uint8, values 0-18 (class index, 0 = background)
    """
    ext = pred_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(pred_path))
        # Normalise axis order: always (H, W, 19)
        if arr.ndim == 3 and arr.shape[0] == NUM_CLASSES:
            arr = arr.transpose(1, 2, 0)        # (19, H, W) -> (H, W, 19)
        # Auto-detect uint8 saved by run_inference.py (values 0-255 = prob * 255)
        if arr.dtype == np.uint8 and arr.max() > 1:
            arr = arr.astype(np.float32) / 255.0   # restore probabilities
        return arr.astype(bool)                     # threshold at 0.5
    elif ext == ".png":
        img = np.array(Image.open(pred_path))       # (H, W), values 0-18
        mask = np.zeros((*img.shape, NUM_CLASSES), dtype=bool)
        for c in range(NUM_CLASSES):
            mask[:, :, c] = img == c
        return mask
    else:
        raise ValueError(f"Unsupported prediction format: {ext}. Use .npy or .png")


# -----------------------------------------------------------------
# IoU computation
# -----------------------------------------------------------------

def compute_iou_per_class(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute per-class IoU between two (H, W, 19) binary arrays.
    Returns an array of shape (19,) with NaN where a class is absent in both.
    """
    iou = np.full(NUM_CLASSES, np.nan)
    for c in range(NUM_CLASSES):
        g = gt[:, :, c]
        p = pred[:, :, c]
        intersection = np.logical_and(g, p).sum()
        union = np.logical_or(g, p).sum()
        if union == 0:
            iou[c] = np.nan   # class absent in this image -- exclude from mean
        else:
            iou[c] = intersection / union
    return iou


# -----------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------

def evaluate(
    data_dir:   str,
    pred_dir:   str,
    split:      str  = "validation",
    ann_dir:    str  = None,
    images_dir: str  = None,
):
    # ----------------------------------------------------------------
    # Resolve all paths robustly (handles Windows backslashes, trailing
    # slashes, relative paths, and leading separators that would make
    # pathlib treat a component as an absolute path).
    # ----------------------------------------------------------------
    pred_dir = Path(pred_dir).resolve()
    data_dir = Path(data_dir).resolve()

    ann_dir_path = (
        Path(ann_dir).resolve() if ann_dir is not None
        else data_dir / "annotations" / split
    )
    img_dir_path = (
        Path(images_dir).resolve() if images_dir is not None
        else data_dir / "images" / split
    )

    # ----------------------------------------------------------------
    # Validate all paths up front with clear diagnostics
    # ----------------------------------------------------------------
    errors = []
    if not pred_dir.exists():
        errors.append(f"  pred_dir not found   : {pred_dir}")
    if not data_dir.exists() and (ann_dir is None or images_dir is None):
        errors.append(f"  data_dir not found   : {data_dir}")
    if not ann_dir_path.exists():
        errors.append(
            f"  Annotations folder not found : {ann_dir_path}\n"
            f"    Expected layout : <data_dir>/annotations/{split}/\n"
            f"    Resolved        : {ann_dir_path}\n"
            f"    Fix: pass the correct --data_dir, or use\n"
            f"         --ann_dir C:/data/dacl10k/annotations/{split}"
        )
    if not img_dir_path.exists():
        errors.append(
            f"  Images folder not found : {img_dir_path}\n"
            f"    Expected layout : <data_dir>/images/{split}/\n"
            f"    Resolved        : {img_dir_path}\n"
            f"    Fix: pass the correct --data_dir, or use\n"
            f"         --images_dir C:/data/dacl10k/images/{split}"
        )
    if errors:
        raise FileNotFoundError("\nPath validation failed:\n" + "\n".join(errors))

    print("=" * 60)
    print(f"  dacl10k mIoU Evaluation  --  split: {split}")
    print("=" * 60)
    print(f"  data_dir     : {data_dir}")
    print(f"  ann_dir      : {ann_dir_path}")
    print(f"  img_dir      : {img_dir_path}")
    print(f"  pred_dir     : {pred_dir}")
    print()

    annotation_files = sorted(ann_dir_path.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation JSON files found in: {ann_dir_path}\n"
            f"  Make sure the folder contains .json annotation files."
        )

    print(f"Found {len(annotation_files)} annotation files.")

    all_iou    = []    # list of (19,) arrays, one per evaluated image
    n_missing  = 0
    n_no_image = 0

    for ann_path in tqdm(annotation_files, desc="Evaluating"):
        stem = ann_path.stem   # e.g. "dacl10k_v2_validation_00001"

        # ---- find corresponding image to get spatial dimensions ----
        img_path = img_dir_path / (stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir_path / (stem + ".png")
        if not img_path.exists():
            tqdm.write(f"  [WARN] Image not found for {stem}, skipping.")
            n_no_image += 1
            continue

        image_shape = np.array(Image.open(img_path)).shape[:2]   # (H, W)

        # ---- ground truth ----
        gt = load_gt_mask_from_annotation(ann_path, image_shape)  # (H, W, 19)

        # ---- prediction: try .npy first, then .png ----
        pred_path = pred_dir / (stem + ".npy")
        if not pred_path.exists():
            pred_path = pred_dir / (stem + ".png")
        if not pred_path.exists():
            tqdm.write(f"  [WARN] Prediction not found for {stem}, skipping.")
            n_missing += 1
            continue

        pred = load_pred_mask(pred_path)   # (H, W, 19)

        # Resize prediction to GT resolution if they differ (also fixes W/H flips)
        if pred.shape[:2] != gt.shape[:2]:
            pred_resized = np.zeros((*gt.shape[:2], NUM_CLASSES), dtype=bool)
            for c in range(NUM_CLASSES):
                ch = Image.fromarray(pred[:, :, c].astype(np.uint8))
                # PIL resize takes (W, H) — gt.shape is (H, W) so reverse it
                ch = ch.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                pred_resized[:, :, c] = np.array(ch).astype(bool)
            pred = pred_resized

        iou = compute_iou_per_class(gt, pred)
        all_iou.append(iou)

    # ----------------------------------------------------------------
    # Summarise skipped files
    # ----------------------------------------------------------------
    if n_missing > 0:
        print(f"\n  WARNING: {n_missing} predictions were missing and skipped.")
        print(f"  Missing images are scored as IoU=0 on CodaLab, not skipped.")
    if n_no_image > 0:
        print(f"  WARNING: {n_no_image} source images were not found.")

    if not all_iou:
        print("\nNo predictions were evaluated. Check your directory paths.")
        return None

    all_iou = np.stack(all_iou, axis=0)   # (N, 19)

    # Compute mean IoU per class (NaN = class absent across all evaluated images)
    class_iou = np.nanmean(all_iou, axis=0)
    miou = np.nanmean(class_iou)

    # ----------------------------------------------------------------
    # Print results table
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Results  --  split: {split}  ({len(all_iou)} images evaluated)")
    print("=" * 60)
    print(f"  {'Class':<20} {'IoU':>8}")
    print("  " + "-" * 30)
    for name, iou_val in zip(CLASS_NAMES, class_iou):
        if np.isnan(iou_val):
            val_str = "     N/A  (no samples)"
        else:
            val_str = f"{iou_val:.4f}"
        print(f"  {name:<20} {val_str}")
    print("  " + "-" * 30)
    print(f"  {'mIoU':<20} {miou:.4f}")
    print("=" * 60)

    print("\nReference mIoU scores:")
    print("  Paper baseline (best)  : 0.4200")
    print("  Challenge winner       : 0.4900  (MatthieuPaques)")
    print("  Challenge 2nd place    : 0.3990  (jfltzngr)")
    print("  Challenge 3rd place    : 0.2060  (XLK)")

    # ----------------------------------------------------------------
    # Save results to JSON next to pred_dir
    # ----------------------------------------------------------------
    results = {
        "split": split,
        "num_images_evaluated": len(all_iou),
        "num_predictions_missing": n_missing,
        "mIoU": float(miou),
        "per_class_IoU": {
            name: (float(val) if not np.isnan(val) else None)
            for name, val in zip(CLASS_NAMES, class_iou)
        },
    }
    out_path = pred_dir / f"evaluation_results_{split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return results


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a dacl10k segmentation model using mIoU."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help=(
            "Root directory of the dacl10k dataset. "
            "Must contain images/<split>/ and annotations/<split>/ subdirectories. "
            "e.g.  C:/data/dacl10k"
        )
    )
    parser.add_argument(
        "--pred_dir", required=True,
        help="Directory containing your model's prediction files (.npy or .png)."
    )
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation"],
        help="Dataset split to evaluate on (default: validation)."
    )
    parser.add_argument(
        "--ann_dir", default=None,
        help=(
            "Optional: path directly to the annotations folder, bypassing the "
            "standard <data_dir>/annotations/<split>/ layout. "
            "e.g.  --ann_dir C:/data/dacl10k/annotations/validation"
        )
    )
    parser.add_argument(
        "--images_dir", default=None,
        help=(
            "Optional: path directly to the images folder, bypassing the "
            "standard <data_dir>/images/<split>/ layout. "
            "e.g.  --images_dir C:/data/dacl10k/images/validation"
        )
    )
    args = parser.parse_args()
    evaluate(
        data_dir=args.data_dir,
        pred_dir=args.pred_dir,
        split=args.split,
        ann_dir=args.ann_dir,
        images_dir=args.images_dir,
    )