# train.py
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, Bottleneck, C2PSA, Detect
import torch.nn as nn

# 导入自定义模块
from modules import C3k2_DWR, EAC, C2PSA_EAC, BiFPNDySampleModule, WiseInnerIoULoss

def register_modules():
    """将自定义模块注册到Ultralytics命名空间"""
    from ultralytics.nn.tasks import parse_model
    original_parse = parse_model

    custom_modules = {
        'C3k2_DWR': C3k2_DWR,
        'C2PSA_EAC': C2PSA_EAC,
        'BiFPNDySampleModule': BiFPNDySampleModule,
        'EAC': EAC
    }

    import types
    def parse_model_with_custom(d, ch, verbose=True):
        for k, v in custom_modules.items():
            if k in d:
                d[k] = v
        return original_parse(d, ch, verbose)

    import ultralytics.nn.tasks as tasks
    tasks.parse_model = parse_model_with_custom
    return custom_modules

# 注册模块
register_modules()

# 加载带自定义模块的模型
model = YOLO('model_configs/cebw_yolo11.yaml')

