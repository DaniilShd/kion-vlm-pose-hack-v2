import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules

register_all_modules()

config = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint = "models/stgcnpp_ntu60_xsub.pth"

print("Loading config...")
cfg = Config.fromfile(config)

print("Building model...")
model = MODELS.build(cfg.model)

print("Loading checkpoint...")
load_checkpoint(model, checkpoint, map_location="cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Model ready on:", device)

# skeleton format
N = 1
M = 2
T = 100
V = 17
C = 3

x = torch.randn(N, M, T, V, C).to(device)

print("Input:", x.shape)

with torch.no_grad():
    feats = model.backbone(x)
    output = model.cls_head(feats)

print("Output:", output.shape)
print("Predicted class:", output.argmax(dim=1).item())