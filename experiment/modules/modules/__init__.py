from .c3k2_dwr import C3k2_DWR, DWR, C3K_DWR
from .eac import EAC, C2PSA_EAC
from .bifpn_dysample import BiFPNDySampleModule, DySample
from .wise_inner_iou import WiseInnerIoULoss

__all__ = [
    "C3k2_DWR", "DWR", "C3K_DWR",
    "EAC", "C2PSA_EAC",
    "BiFPNDySampleModule", "DySample",
    "WiseInnerIoULoss"
]

def register_modules():
    """向Ultralytics命名空间注册自定义模块"""
    import ultralytics.nn.modules as nn_modules
    from ultralytics.nn.tasks import parse_model
    import types

    custom_modules = {
        'C3k2_DWR': C3k2_DWR,
        'C2PSA_EAC': C2PSA_EAC,
        'BiFPNDySampleModule': BiFPNDySampleModule,
        'EAC': EAC
    }

    # 将自定义模块注入到nn_modules
    for k, v in custom_modules.items():
        setattr(nn_modules, k, v)

    # 修改parse_model以识别自定义模块
    original_parse = parse_model

    def parse_model_with_custom(d, ch, verbose=True):
        for k, v in custom_modules.items():
            if k in d:
                d[k] = v
        return original_parse(d, ch, verbose)

    import ultralytics.nn.tasks as tasks
    tasks.parse_model = parse_model_with_custom