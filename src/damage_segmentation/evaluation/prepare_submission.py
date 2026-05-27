"""
dacl-challenge CodaLab Submission Preparation Script
======================================================
Packages your model's predictions into the ZIP format required
for submission to the dacl-challenge on CodaLab:
  https://codalab.lisn.upsaclay.fr/competitions/16317

The submission format expected by the competition is:
  submission.zip
  └── <image_stem>.png   (one per test image, single-channel uint8, values 0-18)

Usage:
    python prepare_submission.py \
        --pred_dir  /path/to/your/predictions \
        --data_dir  /path/to/dacl10k \
        --split     testdev \
        --out_zip   submission_testdev.zip

Prediction format accepted (your model outputs):
    - .npy files: shape (H, W, 19) float/bool  [multi-label, recommended]
    - .png files: single-channel uint8, pixel value = class index 0-18

For multi-label predictions from .npy, the script applies a threshold (default 0.5)
and stores the result as an argmax class-index PNG. If you want to submit
a multi-label binary mask per class, adjust the OUTPUT FORMAT section below.

Requirements:
    pip install numpy pillow tqdm
"""

import os
import json
import zipfile
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# -----------------------------------------------------------------
# dacl10k class definitions (19 classes, index = class id)
# -----------------------------------------------------------------
CLASS_NAMES = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust",
    "Rockpocket", "Hollowareas", "Cavity", "Spalling", "Graffiti",
    "Weathering", "Restformwork", "ExposedRebars", "Bearing",
    "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
]
NUM_CLASSES = len(CLASS_NAMES)

# CodaLab expects background = 0, classes = 1-19
# Pixel value N corresponds to CLASS_NAMES[N-1], value 0 = background
BACKGROUND_IDX = 0


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def load_pred_mask(pred_path: Path) -> np.ndarray:
    """
    Load prediction mask.

    Returns a (H, W, 19) boolean array where
    mask[:, :, c] is True wherever class c is predicted.

    Supports:
      .npy  ->  shape (H, W, 19) or (19, H, W)  float / bool
      .png  ->  single-channel uint8, values 0-18 (class index, 0 = background)
    """
    ext = pred_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(pred_path))
        if arr.ndim == 3 and arr.shape[0] == NUM_CLASSES:
            arr = arr.transpose(1, 2, 0)   # (19, H, W) -> (H, W, 19)
        # Auto-detect uint8 (0-255 from run_inference.py) vs float32 vs bool
        if arr.dtype == np.uint8 and arr.max() > 1:
            arr = arr.astype(np.float32) / 255.0  # restore probabilities
        return arr.astype(bool)  # threshold at 0.5 (i.e. >= 128 for uint8)
    elif ext == ".png":
        img = np.array(Image.open(pred_path))  # (H, W)
        mask = np.zeros((*img.shape, NUM_CLASSES), dtype=bool)
        for c in range(NUM_CLASSES):
            mask[:, :, c] = (img == c)
        return mask
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .npy or .png")


def multilabel_to_submission_png(mask: np.ndarray, threshold: float = 0.5) -> Image.Image:
    """
    Convert a (H, W, 19) multi-label mask to a single-channel submission PNG.

    CodaLab expects one PNG per image where each pixel value encodes
    the predicted class (1-indexed: class 0 -> pixel value 1, background -> 0).

    For multi-label pixels (multiple classes predicted), the class with the
    highest confidence / first occurrence is used. Adjust if your use-case
    requires a different tiebreaking strategy.
    """
    H, W, _ = mask.shape
    out = np.zeros((H, W), dtype=np.uint8)  # 0 = background

    # For each pixel, use the first (lowest-index) predicted class
    # You can change axis=-1 argmax logic here for your preferred tiebreaking
    for c in range(NUM_CLASSES - 1, -1, -1):  # reverse so lowest-index wins
        channel = mask[:, :, c]
        if channel.dtype == float:
            channel = channel >= threshold
        out[channel] = c + 1  # 1-indexed: class 0 -> pixel value 1

    return Image.fromarray(out, mode="L")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def prepare_submission(
    pred_dir: str,
    data_dir: str,
    split: str,
    out_zip: str,
    threshold: float = 0.5,
):
    pred_dir  = Path(pred_dir)
    data_dir  = Path(data_dir)
    out_zip   = Path(out_zip)
    img_dir   = data_dir / "images" / split

    # Collect all test image stems (from the dataset image folder)
    image_stems = sorted(
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_stems:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    print(f"Found {len(image_stems)} images in '{split}' split.")
    print(f"Packaging predictions from: {pred_dir}")

    missing, processed = [], 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in tqdm(image_stems, desc="Packing"):
            # Try .npy first, then .png
            pred_path = pred_dir / (stem + ".npy")
            if not pred_path.exists():
                pred_path = pred_dir / (stem + ".png")
            if not pred_path.exists():
                missing.append(stem)
                continue

            mask = load_pred_mask(pred_path)          # (H, W, 19) bool
            png  = multilabel_to_submission_png(mask, threshold)

            # Write PNG directly into ZIP without saving to disk
            import io
            buf = io.BytesIO()
            png.save(buf, format="PNG")
            buf.seek(0)
            zf.writestr(stem + ".png", buf.read())
            processed += 1

    print(f"\n✓ Packed {processed} predictions into: {out_zip}")
    if missing:
        print(f"⚠ Missing predictions for {len(missing)} images:")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more.")
        print("\n  CodaLab will score missing predictions as all-background (IoU=0).")

    # ----------------------------------------------------------------
    # Submission checklist
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CODALAB SUBMISSION CHECKLIST")
    print("=" * 60)
    print(f"  ZIP file   : {out_zip.name}")
    print(f"  File count : {processed}")
    print(f"  Split      : {split}")
    print()
    print("  Steps to submit:")
    print("  1. Go to: https://codalab.lisn.upsaclay.fr/competitions/16317")
    print("  2. Log in (or create a free account).")
    print("  3. Click 'Participate' → 'Submit / View Results'.")
    print("  4. Select phase:")
    print("       'Development'  → uses testdev  (n=1,012)")
    print("       'Testfinal'    → uses testchallenge (n=998)")
    print(f"  5. Upload: {out_zip.name}")
    print("  6. Wait ~5 min for scoring. Results appear under 'Results'.")
    print()
    print("  Reference scores to beat:")
    print("    Paper baseline (best)  : 0.42 mIoU")
    print("    Challenge winner       : 0.49 mIoU  (MatthieuPaques)")
    print("=" * 60)

    return str(out_zip)


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a CodaLab submission ZIP for the dacl-challenge."
    )
    parser.add_argument(
        "--pred_dir", required=True,
        help="Directory containing your model's .npy or .png prediction files."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root directory of the dacl10k dataset (must contain images/<split>/)."
    )
    parser.add_argument(
        "--split", default="testdev",
        choices=["testdev", "testchallenge"],
        help="Test split to submit (default: testdev)."
    )
    parser.add_argument(
        "--out_zip", default="submission.zip",
        help="Output ZIP filename (default: submission.zip)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binarisation threshold for float .npy predictions (default: 0.5)."
    )
    args = parser.parse_args()
    prepare_submission(
        args.pred_dir, args.data_dir, args.split, args.out_zip, args.threshold
    )