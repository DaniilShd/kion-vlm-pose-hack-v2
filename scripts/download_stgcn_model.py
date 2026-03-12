#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для автоматического скачивания модели ST-GCN для распознавания действий
Совместим с MMAction2 и вашим классом STGCNRecognizer
"""

import os
import sys
import requests
import argparse
from tqdm import tqdm
import json
import yaml
from pathlib import Path

# Конфигурация ссылок на модель
MODEL_URLS = {
    'stgcn_ntu60_xsub': {
        'url': 'https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.pth',
        'filename': 'stgcn_ntu60_2d.pth',
        'description': 'ST-GCN модель, обученная на NTU-60 (XSub)'
    },
    'stgcn_ntu60_xview': {
        'url': 'https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_80e_ntu60_xview_keypoint/stgcn_80e_ntu60_xview_keypoint.pth',
        'filename': 'stgcn_ntu60_xview.pth',
        'description': 'ST-GCN модель, обученная на NTU-60 (XView)'
    }
}

CONFIG_URLS = {
    'stgcn_ntu60_xsub': {
        'url': 'https://raw.githubusercontent.com/open-mmlab/mmaction2/master/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py',
        'filename': 'stgcn_ntu60_2d.py',
        'description': 'Конфиг для NTU-60 XSub'
    },
    'stgcn_ntu60_xview': {
        'url': 'https://raw.githubusercontent.com/open-mmlab/mmaction2/master/configs/skeleton/stgcn/stgcn_80e_ntu60_xview_keypoint.py',
        'filename': 'stgcn_ntu60_xview.py',
        'description': 'Конфиг для NTU-60 XView'
    }
}

# Классы NTU-60 для маппинга
NTU60_CLASSES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up",
    "throw", "sit down", "stand up", "clapping", "reading", "writing",
    "tear up paper", "wear jacket", "take off jacket", "wear a shoe",
    "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking", "punching/slapping",
    "shaking head", "bowing", "cross hands in front", "cross hands in back",
    "snap fingers", "salute", "touching head", "touching chest", "touching back",
    "touching neck", "touching stomach", "neck pain", "headache",
    "chest pain", "back pain", "stomach pain", "fan self", "punching wall",
    "stamping", "hopping", "jump up", "make a phone call/answer phone",
    "playing with phone", "typing on a keyboard", "pointing to something",
    "taking a selfie", "check time (from watch)", "rub two hands together",
    "shake fist", "hugging", "handshaking", "high-five", "hand over head",
    "hand in pocket", "sneezing/coughing", "staggering", "falling"
]

def create_directory_structure():
    """Создает необходимую структуру папок"""
    dirs = ['models', 'config', 'data/results', 'logs']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Создана папка: {dir_path}/")
    
    # Создаем пример paths_config.yaml если его нет
    config_path = Path('config/paths_config.yaml')
    if not config_path.exists():
        default_config = {
            'models': 'models/',
            'configs': 'config/',
            'data': 'data/',
            'results': 'data/results/',
            'logs': 'logs/',
            'checkpoints': 'models/checkpoints/'
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"✓ Создан файл конфигурации: {config_path}")

def download_file(url, destination, description=""):
    """
    Скачивает файл с отображением прогресса
    """
    print(f"\n📥 Скачивание {description}...")
    print(f"   Из: {url}")
    print(f"   В: {destination}")
    
    try:
        # Отправляем GET запрос
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Получаем размер файла
        total_size = int(response.headers.get('content-length', 0))
        
        # Создаем директорию назначения если нужно
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Скачиваем с прогресс-баром
        with open(destination, 'wb') as file:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=os.path.basename(destination),
                colour='green'
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    pbar.update(len(data))
        
        # Проверяем размер скачанного файла
        if total_size > 0:
            downloaded_size = os.path.getsize(destination)
            if downloaded_size != total_size:
                print(f"⚠️  Предупреждение: Размер файла {downloaded_size} не совпадает с ожидаемым {total_size}")
            else:
                print(f"✅ Файл успешно скачан! ({downloaded_size/1024/1024:.2f} MB)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при скачивании: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

def verify_files():
    """Проверяет наличие всех необходимых файлов"""
    print("\n🔍 Проверка файлов...")
    
    required_files = [
        ('models/stgcn_ntu60_2d.pth', 'Модель ST-GCN'),
        ('config/stgcn_ntu60_2d.py', 'Конфигурационный файл'),
        ('config/paths_config.yaml', 'Конфигурация путей')
    ]
    
    all_ok = True
    for file_path, description in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # в MB
            print(f"✅ {description}: {file_path} ({size:.2f} MB)")
        else:
            print(f"❌ {description}: {file_path} не найден")
            all_ok = False
    
    return all_ok

def create_test_script():
    """Создает тестовый скрипт для проверки модели"""
    test_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
Тестовый скрипт для проверки загруженной модели ST-GCN
\"\"\"

import torch
import sys
from pathlib import Path

# Добавляем путь к вашему скрипту
sys.path.append('.')

try:
    from your_stgcn_script import STGCNRecognizer
    
    print("\\n🚀 Инициализация ST-GCN Recognizer...")
    
    # Проверяем доступность CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📊 Используется устройство: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Инициализируем распознаватель
    recognizer = STGCNRecognizer()
    print("✅ Модель успешно загружена!")
    
    # Проверяем структуру
    print("\\n📋 Информация о модели:")
    print(f"   Тип: {type(recognizer.model).__name__}")
    print(f"   Режим: {'eval' if not recognizer.model.training else 'train'}")
    
    # Проверяем маппинг классов
    print(f"\\n🗂️ Маппинг действий:")
    unique_actions = set(recognizer.action_map.values())
    print(f"   Всего классов: {len(unique_actions)}")
    print(f"   Классы: {', '.join(sorted(unique_actions))}")
    
    print("\\n✨ Модель готова к использованию!")
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("   Убедитесь, что ваш скрипт с классом STGCNRecognizer доступен")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")
"""
    
    with open('test_model.py', 'w') as f:
        f.write(test_script)
    print("✅ Создан тестовый скрипт: test_model.py")

def create_sample_paths_config():
    """Создает пример конфигурации путей"""
    config = {
        'models': 'models/',
        'configs': 'config/',
        'data': 'data/',
        'results': 'data/results/',
        'logs': 'logs/',
        'checkpoints': 'models/checkpoints/',
        'mmaction2_configs': 'config/mmaction2/'
    }
    
    with open('config/paths_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("✅ Обновлен config/paths_config.yaml")

def main():
    parser = argparse.ArgumentParser(description='Скачивание ST-GCN модели для распознавания действий')
    parser.add_argument('--model', type=str, default='stgcn_ntu60_xsub',
                        choices=['stgcn_ntu60_xsub', 'stgcn_ntu60_xview'],
                        help='Какую модель скачать')
    parser.add_argument('--no-config', action='store_true',
                        help='Не скачивать конфигурационный файл')
    parser.add_argument('--force', action='store_true',
                        help='Принудительно перезаписать существующие файлы')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 Скачивание модели ST-GCN для распознавания действий")
    print("=" * 60)
    
    # Создаем структуру папок
    create_directory_structure()
    
    model_info = MODEL_URLS[args.model]
    config_info = CONFIG_URLS[args.model]
    
    # Пути к файлам
    model_path = os.path.join('models', model_info['filename'])
    config_path = os.path.join('config', config_info['filename'])
    
    # Проверяем существующие файлы
    if os.path.exists(model_path) and not args.force:
        print(f"\n⚠️  Файл {model_path} уже существует.")
        response = input("Перезаписать? (y/N): ")
        if response.lower() != 'y':
            print("Скачивание модели пропущено.")
        else:
            download_file(model_info['url'], model_path, model_info['description'])
    else:
        download_file(model_info['url'], model_path, model_info['description'])
    
    # Скачиваем конфиг
    if not args.no_config:
        if os.path.exists(config_path) and not args.force:
            print(f"\n⚠️  Файл {config_path} уже существует.")
            response = input("Перезаписать? (y/N): ")
            if response.lower() != 'y':
                print("Скачивание конфига пропущено.")
            else:
                download_file(config_info['url'], config_path, config_info['description'])
        else:
            download_file(config_info['url'], config_path, config_info['description'])
    
    # Создаем тестовый скрипт
    create_test_script()
    
    # Создаем/обновляем конфигурацию путей
    create_sample_paths_config()
    
    # Проверяем результаты
    print("\n" + "=" * 60)
    print("📊 Итоговая проверка:")
    print("=" * 60)
    
    if verify_files():
        print("\n✅ Все необходимые файлы на месте!")
        print("\n🚀 Теперь вы можете:")
        print("   1. Запустить тестовый скрипт: python test_model.py")
        print("   2. Использовать ваш основной скрипт для анализа видео")
        print("\n📝 Пример использования:")
        print("   from your_stgcn_script import STGCNRecognizer")
        print("   recognizer = STGCNRecognizer()")
        print("   result = recognizer.analyze_video('your_video_name')")
    else:
        print("\n⚠️  Некоторые файлы отсутствуют. Проверьте ошибки выше.")
    
    print("\n📌 Примечания:")
    print(f"   - Модель: {model_info['description']}")
    print(f"   - Классов NTU-60: {len(NTU60_CLASSES)}")
    print(f"   - Размер модели: ~{70} MB")
    print("   - Для работы требуется установленный MMAction2")

if __name__ == "__main__":
    main()