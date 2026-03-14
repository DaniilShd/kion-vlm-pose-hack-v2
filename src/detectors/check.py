#!/usr/bin/env python3
import torch
from mmaction.apis import init_recognizer

print("1. Проверка импортов...")
print("   ✅ Импорты успешны")

print("\n2. Проверка CUDA...")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

print("\n3. Загрузка предобученной модели (TSN на Kinetics-400)...")
# Используем готовый конфиг из установленного mmaction2
config_file = '/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

try:
    model = init_recognizer(config_file, checkpoint_file, device=device)
    print("   ✅ Модель успешно создана и загружена!")
    print("\n✅ MMAction2 полностью работает и готов к использованию!")
except Exception as e:
    print(f"\n❌ Ошибка при загрузке модели: {e}")