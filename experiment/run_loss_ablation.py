import subprocess

loss_configs = [
    'configs/loss_ablation/iou.yaml',
    'configs/loss_ablation/giou.yaml',
    'configs/loss_ablation/diou.yaml',
    'configs/loss_ablation/ciou.yaml',
    'configs/loss_ablation/siou.yaml',
    'configs/loss_ablation/wiou.yaml',
    'configs/loss_ablation/inner_iou.yaml',
    'configs/loss_ablation/wise_inner_iou.yaml',
]

for cfg in loss_configs:
    subprocess.run(f"python train.py --config {cfg}", shell=True)