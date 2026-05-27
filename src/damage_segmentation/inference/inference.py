import os
import json
import random
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils import load_model, CLASS_LABELS  # assumes these exist in your project

# ---------------------------
# User config
# ---------------------------
IMAGE_DIR = './images/test'      # folder with images
ANNOTATION_DIR = './annotations/validation'  # folder with JSON annotations (same base name)
CHECKPOINT_PATH = './checkpoints/DeepLabV3Plus_mit_b5_e062_miou0.3802_vloss0.1850.pth'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
THRESHOLD = 0.5

# exact colormap provided by you
CLASS_COLORS = {
    "Crack": (255, 0, 0),
    "ACrack": (0, 255, 0),
    "Wetspot": (0, 0, 255),
    "Efflorescence": (255, 255, 0),
    "Rust": (255, 165, 0),
    "Rockpocket": (128, 0, 128),
    "Hollowareas": (0, 255, 255),
    "Cavity": (255, 192, 203),
    "Spalling": (139, 69, 19),
    "Graffiti": (128, 128, 128),
    "Weathering": (0, 128, 128),
    "Restformwork": (50, 205, 50),
    "ExposedRebars": (75, 0, 130),
    "Bearing": (255, 20, 147),
    "EJoint": (0, 191, 255),
    "Drainage": (139, 0, 139),
    "PEquipment": (173, 255, 47),
    "JTape": (220, 20, 60),
    "WConccor": (0, 100, 0)
}

# Ensure class order matches your model / labels
CLASS_LABELS_ORDERED = CLASS_LABELS  # list expected to be length 19
NUM_CLASSES = len(CLASS_LABELS_ORDERED)

# Build a colors list aligned with CLASS_LABELS_ORDERED. If a label missing in CLASS_COLORS, fallback to gray.
label_colors = [CLASS_COLORS.get(lbl, (127, 127, 127)) for lbl in CLASS_LABELS_ORDERED]

# ---------------------------
# Helpers
# ---------------------------
def pick_random_image(image_dir):
    imgs = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        raise RuntimeError(f"No images found in {image_dir}")
    return random.choice(imgs)

def pil_to_tensor(pil_img):
    # returns tensor shape (1, C, H, W)
    to_tensor = transforms.ToTensor()
    return to_tensor(pil_img).unsqueeze(0)

def predict_probs(model, tensor, device):
    tensor = tensor.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        probs = torch.sigmoid(out).cpu().numpy()[0]  # shape: (C, H, W)
    return probs

def parse_json_to_gt(json_path, image_size, class_labels):
    """
    Parse common JSON formats:
    - If file uses 'shapes' with 'points' and 'label' (LabelMe-like), use that.
    - If file uses 'annotations' with 'class' and 'polygon' (custom), use that.
    Returns GT mask of shape (C, H, W) dtype uint8.
    """
    W, H = image_size
    gt = np.zeros((NUM_CLASSES, H, W), dtype=np.uint8)
    if not os.path.exists(json_path):
        return gt

    with open(json_path, 'r') as f:
        data = json.load(f)

    # LabelMe-style: top-level "shapes"
    if "shapes" in data:
        shapes = data.get("shapes", [])
        for s in shapes:
            label = s.get("label")
            pts = s.get("points", [])
            if label not in class_labels or not pts:
                continue
            idx = class_labels.index(label)
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(gt[idx], [pts_arr], 1)
        return gt

    # Alternative style: "annotations" list with {"class":..., "polygon":[...]}
    if "annotations" in data:
        ann = data.get("annotations", [])
        for item in ann:
            label = item.get("class")
            poly = item.get("polygon", [])
            if label not in class_labels or not poly:
                continue
            idx = class_labels.index(label)
            pts_arr = np.array(poly, dtype=np.int32)
            cv2.fillPoly(gt[idx], [pts_arr], 1)
        return gt

    # Fallback: try to infer shapes in root
    # (if none recognized) return empty gt
    return gt

def masks_to_color_overlay(np_masks, orig_image, label_colors, alpha=0.5):
    """
    np_masks: (C, H, W) uint8
    orig_image: PIL.Image RGB
    label_colors: list of (R,G,B) aligned with masks order
    Returns PIL.Image blended overlay (orig blended with colored masks) and a per-class colored mask (composite)
    """
    H, W = orig_image.size[1], orig_image.size[0]
    orig_np = np.array(orig_image).astype(np.uint8)

    # create a color canvas
    color_canvas = np.zeros_like(orig_np, dtype=np.uint8)

    for c in range(np_masks.shape[0]):
        mask = np_masks[c].astype(bool)
        if not mask.any():
            continue
        color = label_colors[c]
        # apply color where mask is True
        color_canvas[mask] = color

    # blend original and color canvas
    blended = (orig_np * (1 - alpha) + color_canvas * alpha).astype(np.uint8)

    # convert to PIL images
    blended_img = Image.fromarray(blended)
    color_mask_img = Image.fromarray(color_canvas)

    return blended_img, color_mask_img

def draw_labels_on_overlay(overlay_pil, np_masks, class_labels):
    """
    Draw class names at centroids of polygons (if visible).
    """
    draw = ImageDraw.Draw(overlay_pil)
    font = ImageFont.load_default()

    for idx, cls in enumerate(class_labels):
        mask = np_masks[idx].astype(np.uint8)
        if mask.sum() == 0:
            continue

        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # draw text (black outline for readability)
            text = cls
            # basic outline
            draw.text((cx-1, cy-1), text, font=font, fill=(0,0,0))
            draw.text((cx+1, cy+1), text, font=font, fill=(0,0,0))
            draw.text((cx, cy), text, font=font, fill=CLASS_COLORS.get(cls, (255,255,255)))

    return overlay_pil

def build_tp_fp_visual(pred_masks, gt_masks):
    """
    pred_masks & gt_masks: (C,H,W) uint8
    Returns a RGB uint8 numpy image where:
      - TP pixels colored green,
      - FP pixels colored red,
      - FN pixels colored yellow (optional),
      - background dark (optional)
    """
    H, W = pred_masks.shape[1], pred_masks.shape[2]
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    # compute combined per-pixel sets across classes
    tp_any = np.zeros((H, W), dtype=np.bool_)
    fp_any = np.zeros((H, W), dtype=np.bool_)
    fn_any = np.zeros((H, W), dtype=np.bool_)

    for c in range(pred_masks.shape[0]):
        p = pred_masks[c].astype(bool)
        g = gt_masks[c].astype(bool)
        tp_any |= (p & g)
        fp_any |= (p & (~g))
        fn_any |= ((~p) & g)

    # color: TP green, FP red, FN yellow, background dark blue
    vis[fp_any] = [255, 0, 0]      # red
    vis[tp_any] = [0, 255, 0]      # green overrides red where both true
    vis[fn_any & (~tp_any)] = [255, 255, 0]  # yellow for FN (if needed)
    # remaining background default black (0,0,0) or dark blue
    return vis

# ---------------------------
# Main runner
# ---------------------------
def run_random_inference_and_save():
    # pick random image
    img_name = pick_random_image(IMAGE_DIR)
    img_path = os.path.join(IMAGE_DIR, img_name)
    json_path = os.path.join(ANNOTATION_DIR, os.path.splitext(img_name)[0] + ".json")

    print("Selected image:", img_path)
    print("Annotation:", json_path)

    # load image (PIL) and tensor
    pil_img = Image.open(img_path).convert("RGB")
    tensor = pil_to_tensor(pil_img)  # shape (1,C,H,W) using original size

    # load model
    model = load_model(CHECKPOINT_PATH, device=DEVICE)

    # predict probabilities
    probs = predict_probs(model, tensor, DEVICE)  # (C, H, W)

    # ensure number of classes matches
    if probs.shape[0] != NUM_CLASSES:
        raise ValueError(f"Model output channels ({probs.shape[0]}) != NUM_CLASSES ({NUM_CLASSES})")

    # threshold to binary masks
    pred_masks = (probs >= THRESHOLD).astype(np.uint8)  # (C,H,W)

    # load GT masks at original resolution
    image_size = pil_img.size  # (W,H)
    gt_masks = parse_json_to_gt(json_path, image_size, CLASS_LABELS_ORDERED)

    # compute per-class TP/FP pixel masks and counts
    tp_masks = pred_masks * gt_masks
    fp_masks = pred_masks * (1 - gt_masks)

    tp_fp_counts = {}
    for idx, cls in enumerate(CLASS_LABELS_ORDERED):
        tp_count = int(tp_masks[idx].sum())
        fp_count = int(fp_masks[idx].sum())
        tp_fp_counts[cls] = {"TP_pixels": tp_count, "FP_pixels": fp_count}

    # save counts as json
    counts_out_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_tp_fp_counts.json")
    with open(counts_out_path, "w") as f:
        json.dump(tp_fp_counts, f, indent=2)
    print("Saved TP/FP counts ->", counts_out_path)

    # create prediction overlay (colored masks blended with original)
    blended_pil, color_mask_pil = masks_to_color_overlay(pred_masks, pil_img, label_colors, alpha=0.5)
    # add labels on blended overlay
    blended_with_labels = draw_labels_on_overlay(blended_pil, pred_masks, CLASS_LABELS_ORDERED)

    # create TP/FP visualization
    tp_fp_vis_np = build_tp_fp_visual(pred_masks, gt_masks)  # numpy HxWx3
    tp_fp_vis_pil = Image.fromarray(tp_fp_vis_np)

    # Build grid: Original | Prediction overlay | TP/FP
    # ensure all three images have same height; they already should (original size)
    orig_np = np.array(pil_img)
    pred_np = np.array(blended_with_labels)
    tpnp = np.array(tp_fp_vis_pil)

    # If sizes mismatch for any reason, resize to original
    H, W = orig_np.shape[0], orig_np.shape[1]
    def ensure_size(arr, H, W):
        if arr.shape[0] != H or arr.shape[1] != W:
            return cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
        return arr

    orig_np = ensure_size(orig_np, H, W)
    pred_np = ensure_size(pred_np, H, W)
    tpnp = ensure_size(tpnp, H, W)

    # concatenate horizontally
    grid = np.concatenate([orig_np, pred_np, tpnp], axis=1)
    grid_pil = Image.fromarray(grid)

    out_grid_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_grid.png")
    grid_pil.save(out_grid_path)
    print("Saved grid ->", out_grid_path)

    # also save individual artifacts if desired
    blended_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_prediction_overlay.png")
    tp_fp_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_tp_fp.png")
    color_mask_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_color_mask.png")

    blended_with_labels.save(blended_path)
    Image.fromarray(tp_fp_vis_np).save(tp_fp_path)
    color_mask_pil.save(color_mask_path)

    print("Saved artifacts:")
    print(" -", blended_path)
    print(" -", tp_fp_path)
    print(" -", color_mask_path)
    print(" -", counts_out_path)
    print(" -", out_grid_path)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    run_random_inference_and_save()
