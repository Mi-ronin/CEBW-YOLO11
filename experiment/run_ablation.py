# run_ablation.py 改进版
import subprocess
import os

seeds = [42, 123,456,789,2024]
configs = [
    'configs/ablation/model1_baseline.yaml',
    'configs/ablation/model2_c3k2_dwr.yaml',
    'configs/ablation/model3_eac.yaml',
    'configs/ablation/model4_bifpn.yaml',
    'configs/ablation/model5_wise_inner_iou.yaml',
    'configs/ablation/model6_c3k2_eac.yaml',
    'configs/ablation/model7_c3k2_bifpn.yaml',
    'configs/ablation/model8_eac_bifpn.yaml',
    'configs/ablation/model9_all_expect_iou.yaml',
    'configs/ablation/model10_full.yaml',
]

for cfg in configs:
    for seed in seeds:
        # 临时修改配置文件中的 seed 字段
        # 方法1：使用 sed 或 yq 修改
        # 方法2：在命令行传入覆盖参数（需要 train.py 支持）
        cmd = f"python train.py --config {cfg} --seed {seed}"
        subprocess.run(cmd, shell=True)