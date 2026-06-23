from ultralytics import YOLO
from modules import WiseInnerIoULoss
import torch

# 1. 加载带自定义模块的模型
model = YOLO('model_configs/cebw_yolo11.yaml')

# 2. 导入损失函数
criteria = WiseInnerIoULoss(ratio=0.8)

# 3. 模型训练
results = model.train(
    data='dataset.yaml',
    epochs=300,
    batch=8,
    device=0,
    optimizer='AdamW',
    lr0=0.01
)

# 4. 模型验证
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.2f}%")

# 5. 模型导出
model.export(format='onnx')