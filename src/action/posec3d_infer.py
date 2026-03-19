import torch
import numpy as np

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose

from mmaction.registry import MODELS
from mmaction.utils import register_all_modules


class PoseC3DInferencer:
    def __init__(self, config_path, checkpoint_path):
        register_all_modules()
        init_default_scope("mmaction")

        self.cfg = Config.fromfile(config_path)

        # 🔥 берём pipeline
        self.pipeline = Compose(self.cfg.test_pipeline)

        self.model = MODELS.build(self.cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location="cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

        print("PoseC3D loaded on", self.device)

    def infer(self, keypoints):
        """
        keypoints: (1,1,T,17,2)
        """

        N, M, T, V, C = keypoints.shape

        # готовим raw dict (как ожидает pipeline)
        data = dict(
            keypoint=keypoints[0],  # (1,T,17,2)
            keypoint_score=np.ones((M, T, V)),
            total_frames=T,
            img_shape=(256, 256),
            original_shape=(256, 256),
            start_index=0,
            modality='Pose'
        )

        # 🔥 ПРОГОН ЧЕРЕЗ PIPELINE
        data = self.pipeline(data)

        # теперь inputs = tensor
        inputs = data['inputs'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inputs, mode='tensor')

        pred = out.argmax(dim=1).item()

        return pred, out.cpu().numpy()