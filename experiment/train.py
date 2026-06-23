# train.py
import argparse
import yaml
import os
import torch
from ultralytics import YOLO
from utils.logger import ExperimentLogger


def train(config_path):
    # 读取配置（配置文件中已包含所有路径）
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 注册自定义模块（必须在加载模型前）
    from modules import register_modules
    register_modules()

    # 创建实验日志器（输出目录为 runs/exp_name）
    logger = ExperimentLogger(config['exp_name'], base_dir='runs')

    # 记录随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    logger.log_seed(seed)

    # 加载模型（model_yaml可以是官方路径或自定义路径）
    model = YOLO(config['model_yaml'])
    if config.get('pretrained'):
        model.load(config['pretrained'])

    # 开始训练（data 使用绝对路径或相对路径，这里直接使用配置中的路径）
    results = model.train(
        data=config['data_yaml'],  # 例如 ./configs/dataset.yaml
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        device=config['device'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        momentum=config['momentum'],
        workers=config['workers'],
        project=logger.project_dir,  # runs/exp_name 的上级目录？需要调整
        name=config['exp_name'],
        exist_ok=True,
        seed=seed,
        **config.get('hyperparams', {})
    )

    # 评估最佳模型
    from utils.evaluate import evaluate_model, save_predictions_json
    best_model_path = os.path.join(logger.project_dir, config['exp_name'], 'weights', 'best.pt')
    metrics = evaluate_model(
        model_path=best_model_path,
        data_yaml=config['data_yaml'],
        iou_threshold=0.5
    )
    logger.save_metrics(metrics)

    # 保存预测JSON
    pred_json_path = os.path.join(logger.exp_dir, 'predictions.json')
    save_predictions_json(metrics['predictions'], metrics['groundtruths'], pred_json_path)

    logger.close()
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    args = parser.parse_args()
    train(args.config)