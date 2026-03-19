from src.action.posec3d_preprocess import PoseC3DPreprocessor
from src.action.posec3d_infer import PoseC3DInferencer

# пути
NPY_PATH = "data/results/fighting_keypoints.npy"

CONFIG = "configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
CHECKPOINT = "models/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.pth"

# preprocess
prep = PoseC3DPreprocessor()
pose_input = prep.process(NPY_PATH)

print("Input shape:", pose_input.shape)

# inference
model = PoseC3DInferencer(CONFIG, CHECKPOINT)
pred, logits = model.infer(pose_input)

print("Predicted class:", pred)