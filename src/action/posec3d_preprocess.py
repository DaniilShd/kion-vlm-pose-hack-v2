import numpy as np


class PoseC3DPreprocessor:
    def __init__(self, num_frames=48):
        self.num_frames = num_frames

    def load(self, npy_path):
        return np.load(npy_path, allow_pickle=True)

    def extract_single_person(self, data):
        frames = []

        for frame in data:
            kps = frame['keypoints']  # (num_people, 17, 2)

            if len(kps) == 0:
                continue

            frames.append(kps[0])  # берём первого человека

        return frames

    def pad_or_trim(self, frames):
        if len(frames) == 0:
            raise ValueError("Нет keypoints")

        if len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        else:
            frames = frames[:self.num_frames]

        return np.array(frames)  # (T, V, C)

    def normalize(self, frames):
        # YOLO даёт уже xyn (0..1), но центрируем
        frames = frames - 0.5
        return frames

    def process(self, npy_path):
        data = self.load(npy_path)

        frames = self.extract_single_person(data)
        frames = self.pad_or_trim(frames)
        frames = self.normalize(frames)

        # (1,1,T,V,C)
        frames = frames[np.newaxis, np.newaxis, ...]

        return frames