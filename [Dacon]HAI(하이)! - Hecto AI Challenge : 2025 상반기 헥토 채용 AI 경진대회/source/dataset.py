import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class Cutout:
    def __init__(self, length=16, probability=0.3):
        self.length = length
        self.probability = probability
    
    def __call__(self, img):
        if np.random.random() > self.probability:
            return img
        
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

class CarDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.labels = self.df.iloc[:, 2:].values.argmax(axis=1)
        self.image_paths = self.df['img_path'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return (image, self.labels[idx]) if not self.is_test else image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 에러 발생 시 임시 이미지 반환
            dummy_image = torch.zeros(3, 300, 300)
            return (dummy_image, 0) if not self.is_test else dummy_image

def get_train_transform():
    """학습용 데이터 변환"""
    return transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.RandomErasing(p=1.0, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random')
        ], p=0.25),
        transforms.RandomApply([
            Cutout(length=20, probability=1.0)
        ], p=0.15),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """검증용 데이터 변환"""
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_optimized_batch_size(gpu_count, base_batch_size=32):
    """GPU 메모리에 최적화된 배치 크기 계산"""
    if gpu_count > 1:
        effective_batch_size = base_batch_size * gpu_count
        return effective_batch_size
    else:
        return base_batch_size
