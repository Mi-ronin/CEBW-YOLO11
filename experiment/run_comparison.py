import subprocess

comparison_configs = [
    'configs/comparison/faster_rcnn.yaml',
    'configs/comparison/yolov5s.yaml',
    'configs/comparison/yolov7_t.yaml',
    'configs/comparison/yolov8n.yaml',
    'configs/comparison/yolov9_t.yaml',
    'configs/comparison/yolov10n.yaml',
    'configs/comparison/yolo11n.yaml',
    'configs/comparison/rtdetr.yaml',
]

for cfg in comparison_configs:
    subprocess.run(f"python train.py --config {cfg}", shell=True)