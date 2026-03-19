#!/bin/bash
set -e

echo "======================================"
echo "Setting up CTR-GCN (MMAction2)"
echo "======================================"

mkdir -p configs/skeleton/ctrgcn
mkdir -p configs/_base_
mkdir -p models

echo "Downloading CTR-GCN config..."

wget -O configs/skeleton/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/projects/ctrgcn/configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py


echo "Downloading base runtime..."

wget -O configs/_base_/default_runtime.py \
https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/_base_/default_runtime.py


echo "Downloading CTR-GCN pretrained weights..."

wget -O models/ctrgcn_ntu60_xsub.pth \
https://download.openmmlab.com/mmaction/v1.0/projects/ctrgcn/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20230308-7aba454e.pth


echo "Checking files..."

echo "Configs:"
ls -R configs/skeleton/ctrgcn

echo "Models:"
ls -lh models

echo "======================================"
echo "CTR-GCN ready ✔"
echo "======================================"