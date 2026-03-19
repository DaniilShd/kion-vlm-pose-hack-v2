import os
import torch

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules

register_all_modules()

CONFIG_PATH = "configs/skeleton/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
CHECKPOINT_PATH = "models/ctrgcn_ntu60_xsub.pth"

print("Loading config...")
cfg = Config.fromfile(CONFIG_PATH)

print("Building model...")
model = MODELS.build(cfg.model)

print("Loading checkpoint...")
load_checkpoint(model, CHECKPOINT_PATH, map_location="cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Model ready on:", device)

# (N, M, T, V, C)
x = torch.randn(1, 2, 100, 17, 3).to(device)

with torch.no_grad():
    output = model(x)

print("Output shape:", output.shape)
print("Prediction:", output.argmax(dim=1).item())