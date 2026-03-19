import torch

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope

from mmaction.registry import MODELS
from mmaction.utils import register_all_modules

# =========================
# Init
# =========================
register_all_modules()
init_default_scope("mmaction")

CONFIG_PATH = "configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
CHECKPOINT_PATH = "models/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.pth"

cfg = Config.fromfile(CONFIG_PATH)

model = MODELS.build(cfg.model)
load_checkpoint(model, CHECKPOINT_PATH, map_location="cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Model ready on:", device)

# =========================
# ✅ CORRECT INPUT
# =========================
N, M, T, V, C = 1, 1, 48, 17, 2

keypoint = torch.randn(N, M, T, V, C).to(device)

print("Input shape:", keypoint.shape)

# =========================
# Forward
# =========================
with torch.no_grad():
    outputs = model(keypoint, mode='tensor')

print("Output shape:", outputs.shape)
print("Predicted class:", outputs.argmax(dim=1).item())