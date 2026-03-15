#!/bin/bash
set -e

echo "======================================"
echo "Setting up ST-GCN (NTU60 XSub)"
echo "======================================"

# Создаём правильную структуру
mkdir -p configs/skeleton/stgcn
mkdir -p configs/_base_
mkdir -p models

echo "Downloading ST-GCN config..."

wget -O configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py

echo "Downloading base runtime config..."

wget -O configs/_base_/default_runtime.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/default_runtime.py

echo "Downloading weights..."

wget -O models/stgcn_ntu60.pth \
https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth

echo "Checking structure..."

ls -R configs
ls -lh models

echo "Done ✔"