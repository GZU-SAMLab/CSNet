import os
from PIL import ImageFilter
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms


class Countgwhd(Dataset):
    def __init__(self, img_path, ann_path, resize_shape):
        self.img_path = img_path
        self.ann_path = pd.read_csv(ann_path)
        self.shape = resize_shape
        self.transform = transforms.Resize([self.shape, self.shape])

    def __getitem__(self, idex):
        # 拼接图片
        img_path = os.path.join(self.img_path + self.ann_path.iloc[idex, 0])
        # tensor类型
        image = Image.open(img_path)
        image = image.convert("RGB")
        TOtensor = transforms.ToTensor()
        image = TOtensor(image) * 255
        label = self.ann_path.iloc[idex, 1]
        image = self.transform(image)
        return image, label

    def __len__(self):
            return len(self.ann_path)



