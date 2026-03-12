# -----------------------------
# Базовый образ NVIDIA PyTorch с CUDA 13 и Python 3.12
# -----------------------------
FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="13.0"

# -----------------------------
# Системные зависимости
# -----------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg git \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Виртуальное окружение
# -----------------------------
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# -----------------------------
# Базовые ML библиотеки
# -----------------------------
RUN pip install --no-cache-dir \
    numpy scipy pandas matplotlib tqdm pillow \
    opencv-python decord einops \
    ultralytics==8.3.101 \
    transformers accelerate sentencepiece safetensors \
    torchmetrics

# -----------------------------
# ST-GCN / Skeleton-based Action Recognition
# -----------------------------
# Используем локальный код ST-GCN, не git-клон
# Предположим, что в проекте есть папка stgcn с исходниками
# RUN pip install -e ./stgcn

# -----------------------------
# Рабочая директория
# -----------------------------
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN pip install -r requirements.txt

# -----------------------------
# Проверка GPU
# -----------------------------
RUN python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Torch version:", torch.__version__)
EOF

# -----------------------------
# Контейнер в режиме “живой”, можно запускать main.py
# -----------------------------
CMD ["tail", "-f", "/dev/null"]