import cv2
import json
import os
import yaml
import time
import logging
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.pose.utils.visualization import draw_skeleton

class PoseDetector:
    def __init__(self):
        # конфиги
        with open("config/pose_config.yaml") as f:
            self.cfg = yaml.safe_load(f)
        with open("config/paths_config.yaml") as f:
            self.paths = yaml.safe_load(f)
        
        # логгер
        log_file = f"{self.paths['logs']}/pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # модель
        model_name = self.cfg['model']['name']
        model_path = os.path.join(self.paths['models'], model_name)
        self.model = YOLO(model_path if os.path.exists(model_path) else model_name)
        self.frame_step = self.cfg['video']['frame_step']
        
        if self.cfg['model']['device'] == 'cuda':
            try:
                self.model.to('cuda')
                print("Модель на GPU")
            except:
                print("CUDA не доступна, использую CPU")
    
    def process_video(self, video_path):
        video_name = Path(video_path).stem
        
        # открываем видео
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nВидео: {total_frames} кадров, {fps:.2f} fps, {video_duration:.2f} сек")
        print(f"Обрабатываем каждый {self.frame_step}-й кадр")
        
        # для записи
        fourcc = cv2.VideoWriter_fourcc(*self.cfg['video']['codec'])
        out_video = os.path.join(self.paths['results'], f"{video_name}_pose.mp4")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        
        # данные для бинарного сохранения
        all_frames = []  # список всех обработанных кадров с ключевыми точками
        frame_num = 0
        processed = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % self.frame_step == 0:
                results = self.model(frame, conf=self.cfg['model']['confidence'], verbose=False)
                
                if results[0].keypoints is not None:
                    kps = results[0].keypoints.xyn.cpu().numpy()
                    
                    # сохраняем ключевые точки в список для бинарного файла
                    frame_keypoints = {
                        'frame': frame_num,
                        'timestamp': frame_num / fps if fps > 0 else 0,
                        'keypoints': kps  # сохраняем как numpy array
                    }
                    all_frames.append(frame_keypoints)
                    
                    frame = draw_skeleton(frame, kps)
                
                processed += 1
                if processed % self.cfg['logging']['print_interval'] == 0:
                    print(f"Обработано {processed} кадров")
            
            writer.write(frame)
            frame_num += 1
        
        # статистика
        processing_time = time.time() - start_time
        proc_fps = processed / processing_time if processing_time > 0 else 0
        effective_fps = proc_fps * self.frame_step
        
        log_data = {
            'video': video_name,
            'total_frames': total_frames,
            'processed': processed,
            'video_duration': round(video_duration, 2),
            'processing_time': round(processing_time, 2),
            'frame_step': self.frame_step,
            'fps': round(fps, 2),
            'proc_fps': round(proc_fps, 2),
            'effective_fps': round(effective_fps, 2),
            'speedup': round(video_duration/processing_time if processing_time>0 else 0, 2)
        }
        
        self.logger.info(f"{video_name}: {log_data}")
        
        print(f"\nГОТОВО")
        print(f"Время обработки: {processing_time:.2f} сек")
        print(f"Длительность видео: {video_duration:.2f} сек")
        print(f"Шаг кадров: {self.frame_step}")
        print(f"Скорость: {proc_fps:.2f} fps")
        print(f"Эффективный FPS: {effective_fps:.2f}")
        print(f"Ускорение: {log_data['speedup']:.2f}x")
        
        # сохраняем ключевые точки в бинарном формате
        npy_path = os.path.join(self.paths['results'], f"{video_name}_keypoints.npy")
        np.save(npy_path, all_frames)  # all_frames как numpy array
        
        # в JSON только метаданные
        json_path = os.path.join(self.paths['results'], f"{video_name}_meta.json")
        with open(json_path, 'w') as f:
            json.dump({
                'video': video_name,
                'fps': fps,
                'frame_step': self.frame_step,
                'processed_frames': processed,
                'keypoints_file': f"{video_name}_keypoints.npy"
            }, f, indent=2)
        
        cap.release()
        writer.release()
        
        print(f"Ключевые точки (бинарный): {npy_path}")
        print(f"Метаданные: {json_path}")
        print(f"Видео: {out_video}")
        
        return npy_path, json_path, out_video

def main():
    detector = PoseDetector()
    input_dir = detector.paths['input_dir']
    
    videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        print(f"Нет видео в {input_dir}")
        return
    
    print(f"\nВидео в папке:")
    for v in videos:
        print(f"  - {v}")
    
    for v in videos:
        print(f"\n--- Обрабатываю {v} ---")
        try:
            detector.process_video(os.path.join(input_dir, v))
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()