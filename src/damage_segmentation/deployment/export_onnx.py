import torch
import segmentation_models_pytorch as smp
from model import SegmentationModel
from utils import load_model
import yaml
import tqdm as tqdm

# -----------------------------
# 1. Load your model
# -----------------------------

with open("config_training_rtx4500.yaml", "r") as f:
    cfg = yaml.safe_load(f)


MODEL_CFG = cfg.get("model", {})
ARCHITECTURE = MODEL_CFG.get("architecture", "DeepLabV3Plus")
ENCODER = MODEL_CFG.get("encoder", "mit_b5")
WEIGHTS = MODEL_CFG.get("weights", "imagenet")
NUM_CLASSES = MODEL_CFG.get("num_classes", 19)
MODEL_PATH = "./checkpoints/DeepLabV3Plus_mit_b5_e055_miou0.3744_vloss0.1836.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = SegmentationModel(arch=ARCHITECTURE, encoder=ENCODER, weights=WEIGHTS, num_classes=NUM_CLASSES)

# Load weights
checkpoint = load_model(MODEL_PATH, device=DEVICE)
model.to(DEVICE)
model.eval()

# -----------------------------
# 2. Create dummy input
# -----------------------------
# Must match training input size
dummy_input = torch.randn(1, 3, 640, 640).to(DEVICE)

# -----------------------------
# 3. Export to ONNX
# -----------------------------
ONNX_PATH = "deeplabv3plus_mitb5.onnx"

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,              # 13 or 17 recommended
    do_constant_folding=True,
    
    input_names=["input"],
    output_names=["output"],
    
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)


print(f"Model exported to {ONNX_PATH}")