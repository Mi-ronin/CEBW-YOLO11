import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transforms = transforms
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png','.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg','.txt').replace('.png','.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, w, h = map(float, parts)
                        boxes.append([cls, xc, yc, w, h])
        labels = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,5))
        # 数据增强等...
        return image, labels