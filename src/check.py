#!/usr/bin/env python
"""Проверка готовности Docker контейнера к работе с ML моделями."""

import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime
import os
import torch
import cv2
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class ContainerHealthCheck:
    def __init__(self, config_path="config/paths_config.yaml"):
        self.results = {}
        self.start_time = datetime.now()
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Загружает конфиг с путями."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except:
            logger.warning("Конфиг не найден, использую пути по умолчанию")
            return {
                'data_dir': './data', 'input_dir': './data/input', 'results': './data/results',
                'uploads': './data/uploads', 'logs': './logs', 'cache': './cache', 'models': './models',
                'pose_detector': 'models/yolov8n-pose.pt', 'glm4v_path': 'models/glm-4v'
            }
    
    def run_all_checks(self):
        """Запускает все проверки."""
        print("\n" + "="*50)
        print("ПРОВЕРКА КОНТЕЙНЕРА")
        print("="*50)
        
        # Инфо
        logger.info(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
        
        # Проверки
        self._check_directories()
        self._check_cuda()
        self._check_memory()
        self._check_disk_space()
        
        # Итоги
        print("\n" + "="*50)
        print("ИТОГИ:")
        all_ok = True
        for name, ok in self.results.items():
            print(f"[{'✓' if ok else '✗'}] {name}")
            if not ok: all_ok = False
        
        print(f"\nВремя: {(datetime.now()-self.start_time).total_seconds():.1f} сек")
        print("✅ КОНТЕЙНЕР ГОТОВ" if all_ok else "❌ ЕСТЬ ПРОБЛЕМЫ")
        return all_ok
    
    def _check_directories(self):
        """Проверяет директории из конфига."""
        logger.info("Проверка директорий...")
        ok = True
        
        for name, path_str in self.config.items():
            if not isinstance(path_str, str): 
                continue
            
            # Пропускаем проверку для путей к файлам моделей
            if name in ['pose_detector', 'glm4v_path']:
                # Проверяем только родительскую директорию
                path = Path(path_str).parent
            else:
                path = Path(path_str)
            
            try:
                if not path.exists():
                    logger.warning(f"  {name}: {path} не существует")
                    ok = False
                elif not os.access(path, os.W_OK):
                    logger.warning(f"  {name}: нет прав на запись")
                    ok = False
                else:
                    logger.info(f"  ✓ {name}: {path}")
            except:
                logger.warning(f"  {name}: ошибка проверки")
                ok = False
        
        self.results['Директории'] = ok
    
    def _check_cuda(self):
        """Проверяет GPU."""
        logger.info("Проверка CUDA...")
        
        if not torch.cuda.is_available():
            logger.warning("  CUDA не найдена, будет CPU")
            self.results['CUDA/GPU'] = False
            return
        
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  ✓ GPU: {gpu}, {mem:.1f} GB")
        
        # Простые тесты без обратного распространения
        try:
            # Простое умножение матриц
            a = torch.randn(100, 100).cuda()
            b = torch.randn(100, 100).cuda()
            c = a @ b
            logger.info(f"  ✓ Умножение матриц работает: {c.shape}")
            
            # Проверка что тензоры на GPU
            logger.info(f"  ✓ Тензоры на GPU: {a.device}")
            
            self.results['CUDA/GPU'] = True
        except Exception as e:
            logger.error(f"  Ошибка: {e}")
            self.results['CUDA/GPU'] = False
    
    def _check_memory(self):
        """Проверяет память."""
        logger.info("Проверка памяти...")
        
        mem = psutil.virtual_memory()
        logger.info(f"  RAM: {mem.available/1e9:.1f} ГБ свободно")
        
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(f"  GPU: {free/1e9:.1f} ГБ свободно из {total/1e9:.1f} ГБ")
            if free < 2e9:
                logger.warning("  Мало GPU памяти!")
        
        self.results['Память'] = True
    
    def _check_disk_space(self):
        """Проверяет место на диске."""
        logger.info("Проверка места на диске...")
        
        for key in ['models', 'data_dir', 'logs']:
            if key in self.config:
                path = Path(self.config[key])
                if path.exists():
                    free = psutil.disk_usage(path).free / 1e9
                    logger.info(f"  {key}: {free:.1f} ГБ свободно")
                    if free < 5:
                        logger.warning(f"  Мало места в {key}!")
        
        self.results['Диск'] = True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/paths_config.yaml')
    args = parser.parse_args()
    
    checker = ContainerHealthCheck(args.config)
    return 0 if checker.run_all_checks() else 1

if __name__ == "__main__":
    sys.exit(main())