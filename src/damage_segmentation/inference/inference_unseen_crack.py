import os
import cv2
import random
import torch
import numpy as np
from torchvision import transforms
from utils import load_model, CLASS_LABELS

# ------------------- CONFIG -------------------
TEST_IMAGE_DIR = './images/test'
CHECKPOINT_PATH = './checkpoints/DeepLabV3Plus_mit_b5_e062_miou0.3802_vloss0.1850.pth'

NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE_SIZE = 1024
OVERLAP = 0.25
USE_TTA = True
TTA_MODES = ['none', 'hflip', 'vflip', 'hvflip']

DEFAULT_THRESHOLD = 0.5
CLASS_THRESHOLDS = [DEFAULT_THRESHOLD] * NUM_CLASSES

OUTPUT_DIR = "./test_inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette (your CLASS_COLORS converted to list order)
label_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 165, 0), (128, 0, 128), (0, 255, 255), (255, 192, 203),
    (139, 69, 19), (128, 128, 128), (0, 128, 128), (50, 205, 50),
    (75, 0, 130), (255, 20, 147), (0, 191, 255), (139, 0, 139),
    (173, 255, 47), (220, 20, 60), (0, 100, 0)
]


# ------------------- CRACK CONFIG -------------------
CRACK_CLASS_NAME = "Crack"   # must match CLASS_LABELS
CRACK_CLASS_IDX = CLASS_LABELS.index(CRACK_CLASS_NAME)

CRACK_COLOR = (255, 0, 0)    # RED in RGB
CRACK_THRESHOLD = 0.5

# ------------------- HELPERS -------------------
def read_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def overlay_crack_prediction(crack_mask, image, alpha=0.5):
    """
    crack_mask: (H, W) binary mask
    image: RGB image
    """
    overlay = image.copy()

    crack_mask = crack_mask.astype(np.uint8)

    if crack_mask.sum() > 0:
        contours, _ = cv2.findContours(
            crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            cv2.drawContours(overlay, [cnt], -1, CRACK_COLOR, -1)

    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


# ------------------- TTA + TILE -------------------
def apply_tta_and_predict(model, patch):
    tensor_transform = transforms.ToTensor()
    preds = []

    for mode in (TTA_MODES if USE_TTA else ['none']):
        if mode == 'none':
            img_mod = patch
        elif mode == 'hflip':
            img_mod = np.flip(patch, axis=1).copy()
        elif mode == 'vflip':
            img_mod = np.flip(patch, axis=0).copy()
        elif mode == 'hvflip':
            img_mod = np.flip(np.flip(patch, axis=0), axis=1).copy()
        else:
            img_mod = patch

        inp = tensor_transform(img_mod).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)
            probs = torch.sigmoid(out).squeeze(0).cpu().numpy()

        if mode == 'hflip':
            probs = np.flip(probs, axis=2)
        elif mode == 'vflip':
            probs = np.flip(probs, axis=1)
        elif mode == 'hvflip':
            probs = np.flip(np.flip(probs, axis=1), axis=2)

        preds.append(probs)

    return np.mean(preds, axis=0)

def sliding_window_inference(image, model, tile_size=TILE_SIZE, overlap=OVERLAP):
    H, W, _ = image.shape
    stride = int(tile_size * (1 - overlap))

    prob_map = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)

            patch = image[y:y2, x:x2]
            ph, pw = patch.shape[:2]

            pad_h = tile_size - ph
            pad_w = tile_size - pw
            if pad_h > 0 or pad_w > 0:
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            probs_full = apply_tta_and_predict(model, patch)
            probs = probs_full[:, :ph, :pw]

            prob_map[:, y:y2, x:x2] += probs
            count_map[y:y2, x:x2] += 1

    count_map[count_map == 0] = 1
    return prob_map / count_map[np.newaxis, :, :]
def infer_test_image(model, img_path, save=True):
    original = read_image_rgb(img_path)
    H, W = original.shape[:2]

    prob_map = sliding_window_inference(original, model)

    # ---- Extract ONLY crack channel ----
    crack_prob = prob_map[CRACK_CLASS_IDX]
    crack_binary = (crack_prob >= CRACK_THRESHOLD).astype(np.uint8)

    overlay = overlay_crack_prediction(crack_binary, original)

    # ---- Ensure saved overlay is same size as original ----
    overlay_resized = cv2.resize(overlay, (W, H), interpolation=cv2.INTER_LINEAR)

    if save:
        base = os.path.splitext(os.path.basename(img_path))[0]

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{base}_crack_overlay.png"),
            cv2.cvtColor(overlay_resized, cv2.COLOR_RGB2BGR)
        )

        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"{base}_crack_probs.npz"),
            crack_prob=crack_prob
        )

    print(f"[DONE] Saved crack-only overlay (original size) for {img_path}")

    return {
        "crack_prob": crack_prob,
        "crack_mask": crack_binary,
        "overlay": overlay_resized
    }


# ------------------- RANDOM TEST IMAGE -------------------
def run_inference_random_test():
    model = load_model(CHECKPOINT_PATH, device=DEVICE).to(DEVICE)
    model.eval()

    imgs = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not imgs:
        raise RuntimeError("No test images found!")

    img_name = random.choice(imgs)
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)

    print("Selected test image:", img_name)
    infer_test_image(model, img_path, save=True)

if __name__ == "__main__":
    run_inference_random_test()
