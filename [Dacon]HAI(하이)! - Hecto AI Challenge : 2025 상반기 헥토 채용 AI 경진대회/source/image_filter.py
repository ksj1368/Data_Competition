import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ImageFilter:
    """�̹��� ���͸��� ����ϴ� Ŭ����"""
    
    def __init__(self, model_name: str = 'resnet50', max_workers: int = 8):
        self.model_name = model_name
        self.max_workers = max_workers
        self.model = None
        self.preprocess = None
        self.car_classes = [468, 475, 479, 555, 656, 569, 575, 609, 675, 705, 717, 734, 751, 779, 817, 864, 867, 874]
        self._setup_model()
    
    def _setup_model(self) -> None:
        """�𵨰� ��ó�� ���������� �ʱ�ȭ"""
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"�������� �ʴ� ��: {self.model_name}")
            
        self.model.eval()
        
        # ��ó�� ���������� ����
        self.preprocess = transforms.Compose([
            transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"{self.model_name} ���� �ε�Ǿ����ϴ�.")
    
    def classify_single_image(self, img_path: str, threshold: float = 0.1) -> Tuple[int, float]:
        """���� �̹��� �з�"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs[0], dim=0)
            
            # �ڵ��� Ŭ�������� Ȯ�� �հ� ���
            car_prob = probs[self.car_classes].sum().item()
            
            # �Ӱ谪 �������� �� ���� (0: �ڵ���, 1: �ڵ��� �ƴ�)
            label = 0 if car_prob > threshold else 1
            
            return label, car_prob
            
        except Exception as e:
            print(f"�̹��� ó�� ���� {img_path}: {e}")
            return 1, 0.0  # ���� �߻� �� �ڵ��� �ƴ����� �з�
    
    def classify_images_parallel(self, img_paths: List[str], threshold: float = 0.1) -> List[Tuple[int, float]]:
        """���� ó���� ���� �̹��� �з�"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # �۾� ����
            future_to_path = {
                executor.submit(self.classify_single_image, img_path, threshold): img_path 
                for img_path in img_paths
            }
            
            # ��� ����
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"���� ó�� ���� {img_path}: {e}")
                    results.append((1, 0.0))
        
        return results
    
    def filter_dataframe(self, df: pd.DataFrame, threshold: float = 0.1, 
                        img_path_column: str = 'img_path') -> pd.DataFrame:
        """�����������ӿ� ���͸� ��� �߰�"""
        if img_path_column not in df.columns:
            raise ValueError(f"�÷� '{img_path_column}'�� �����������ӿ� �����ϴ�.")
        
        print("�̹��� ���͸��� �����մϴ�...")
        img_paths = df[img_path_column].tolist()
        
        # ���� ó���� �з� ����
        results = self.classify_images_parallel(img_paths, threshold)
        
        # ����� �����������ӿ� �߰�
        df_copy = df.copy()
        df_copy['label'], df_copy['prob'] = zip(*results)
        
        # ��� ���
        car_count = (df_copy['label'] == 0).sum()
        non_car_count = (df_copy['label'] == 1).sum()
        
        print(f"���͸� �Ϸ�:")
        print(f"- �ڵ��� �̹���: {car_count}��")
        print(f"- �ڵ��� �ƴ�: {non_car_count}��")
        print(f"- ��ü �̹���: {len(df_copy)}��")
        
        return df_copy
    
    def update_car_classes(self, new_car_classes: List[int]) -> None:
        """�ڵ��� Ŭ���� ��� ������Ʈ"""
        self.car_classes = new_car_classes
        print(f"�ڵ��� Ŭ������ ������Ʈ�Ǿ����ϴ�: {len(self.car_classes)}�� Ŭ����")
    
    def get_filter_statistics(self, df: pd.DataFrame) -> dict:
        """���͸� ��� ��ȯ"""
        if 'label' not in df.columns or 'prob' not in df.columns:
            raise ValueError("���͸��� ������� ���� �������������Դϴ�.")
        
        stats = {
            'total_images': len(df),
            'car_images': (df['label'] == 0).sum(),
            'non_car_images': (df['label'] == 1).sum(),
            'avg_car_probability': df[df['label'] == 0]['prob'].mean() if (df['label'] == 0).any() else 0,
            'avg_non_car_probability': df[df['label'] == 1]['prob'].mean() if (df['label'] == 1).any() else 0
        }
        
        return stats
