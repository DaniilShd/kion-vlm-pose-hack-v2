import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmaction.apis import inference_recognizer

register_all_modules()

config = "configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
checkpoint = "models/stgcn_ntu60.pth"

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

# ST-GCN input format
# (N, C, T, V, M)

N = 1
M = 2 
T = 100    # frames
V = 17     # joints (COCO)     # persons
C = 3      # x,y,score


x = torch.randn(N, M, T, V, C).to(device)

print("Input:", x.shape)

data = dict(inputs=x)

with torch.no_grad():
    feats = model.backbone(x)
    output = model.cls_head(feats)

print("Output:", output.shape)
print("Classes:", output.argmax(dim=1).item())