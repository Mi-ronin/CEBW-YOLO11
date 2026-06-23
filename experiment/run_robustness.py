"""
run_robustness.py
运行所有鲁棒性测试，生成表10所需数据
"""
import argparse
import yaml
import os
import sys

sys.path.append('.')  # 确保可以导入utils

from utils.robustness import run_all_robustness_tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/robustness_config.yaml',
                        help='鲁棒性测试配置文件')
    parser.add_argument('--model', type=str, help='模型路径（覆盖配置文件）')
    parser.add_argument('--data', type=str, help='数据集配置路径')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_path = args.model or config['model_path']
    data_yaml = args.data or config['data_yaml']
    device = args.device or config['device']
    output_dir = config['output_dir']

    # 验证模型文件存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_all_robustness_tests(model_path, data_yaml, output_dir, device)


if __name__ == '__main__':
    main()