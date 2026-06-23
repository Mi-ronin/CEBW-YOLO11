import os
import json
import yaml


class ExperimentLogger:
    def __init__(self, exp_name, base_dir='runs'):
        self.exp_name = exp_name
        self.base_dir = base_dir
        self.project_dir = base_dir  # 与ultralytics的project参数对应
        self.exp_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.exp_dir, 'metrics.json')
        self.seed_file = os.path.join(self.exp_dir, 'random_seed.txt')

    def log_seed(self, seed):
        with open(self.seed_file, 'w') as f:
            f.write(str(seed))

    def save_metrics(self, metrics):
        # 过滤过大字段
        safe_metrics = {k: v for k, v in metrics.items() if k not in ['predictions', 'groundtruths']}
        with open(self.metrics_file, 'w') as f:
            json.dump(safe_metrics, f, indent=2)

    def save_config(self, config):
        config_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

    def close(self):
        pass