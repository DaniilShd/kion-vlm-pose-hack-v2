#!/bin/bash
set -e

echo "======================================"
echo "Setting up PoseC3D (NTU60 XSub)"
echo "======================================"

# Создаём структуру
mkdir -p configs/skeleton/posec3d
mkdir -p configs/_base_
mkdir -p models

echo "Downloading PoseC3D config..."

wget -O configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py

echo "Downloading base runtime config..."

wget -O configs/_base_/default_runtime.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/default_runtime.py

echo "Downloading PoseC3D weights..."

wget -O models/posec3d_ntu60.pth \
https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth

echo "Checking structure..."

ls -R configs
ls -lh models

echo "Done ✔"