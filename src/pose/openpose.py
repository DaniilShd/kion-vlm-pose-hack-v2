import os
import sys
import cv2
import numpy as np
import torch

sys.path.append('/path/to/openpose/build/python')  # путь к pyopenpose
import pyopenpose as op

from posec3d_infer import PoseC3DInferencer

# -------------------
# Конфиги PoseC3D
# -------------------
CONFIG_PATH = "configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
CHECKPOINT_PATH = "models/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.pth"

pose_model = PoseC3DInferencer(CONFIG_PATH, CHECKPOINT_PATH)

# -------------------
# Настройка OpenPose
# -------------------
params = dict()
params["model_folder"] = "/path/to/openpose/models/"
params["net_resolution"] = "368x368"
params["number_people_max"] = 1

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# -------------------
# Обрабатываем видео
# -------------------
video_path = "data/input/fighting.mp4"
cap = cv2.VideoCapture(video_path)

keypoints_all = []
frame_step = 1
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % frame_step == 0:
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        if datum.poseKeypoints is not None:
            # shape = (num_people, 25, 3)
            kps = datum.poseKeypoints[0][:17, :2]  # берем первые 17 точек
            kps = kps[np.newaxis, np.newaxis, :, :, :]  # (1,1,T,V,C)
            scores = datum.poseKeypoints[0][:17, 2:3]
            
            # передаём в PoseC3D
            pred, logits = pose_model.infer(kps)
            print(f"Frame {frame_num}: Predicted action {pred}")

    frame_num += 1

cap.release()