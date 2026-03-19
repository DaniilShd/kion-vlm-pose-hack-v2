#!/bin/bash
set -e

echo "======================================"
echo "Setting up PoseC3D (NTU60 XSub)"
echo "======================================"

# Создаём ВСЮ нужную структуру
mkdir -p configs/skeleton/posec3d
mkdir -p configs/_base_/models
mkdir -p configs/_base_/schedules
mkdir -p configs/_base_
mkdir -p models

echo "Downloading PoseC3D config..."
wget -O configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py

echo "Downloading base configs..."

wget -O configs/_base_/default_runtime.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/default_runtime.py

wget -O configs/_base_/models/slowonly_r50.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/models/slowonly_r50.py

wget -O configs/_base_/schedules/sgd_50e.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/schedules/sgd_50e.py

echo "Downloading PoseC3D weights..."
wget -O models/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.pth \
https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth

echo "Done ✔"