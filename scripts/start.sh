# Переходим в /app (хотя вы там уже должны быть)
cd /app

# Создаем папку для весов
mkdir -p weights

# Скачиваем конфиг в папку с проектом (не в mmaction2, чтобы не было проблем с правами)
wget -O stgcn_ntu60_config.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py

# Скачиваем веса в папку weights
wget -O weights/stgcn_ntu60_joint.pth https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth

# Проверяем, что скачалось
ls -la stgcn_ntu60_config.py weights/