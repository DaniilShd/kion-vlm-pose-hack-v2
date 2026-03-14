import os
import subprocess

def download_with_wget(url, filename):
    """Скачивание через wget"""
    try:
        subprocess.run(['wget', '-O', filename, url], check=True)
        return True
    except:
        return False

# Использование:
checkpoint_url = "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221207-221aef19.pth"
if not os.path.exists(checkpoint_url):
    print("Скачивание через wget...")
    download_with_wget(checkpoint_url, checkpoint_url)