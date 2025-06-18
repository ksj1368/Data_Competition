import numpy as np
import pandas as pd
import os
import warnings
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """������ ��ó���� ����ϴ� Ŭ����"""
    
    def __init__(self, train_dir: str):
        self.train_dir = train_dir
        self.class_folders = []
        self.image_data = []
        
    def get_class_folders(self) -> List[str]:
        """Ŭ���� �������� �������� ����"""
        self.class_folders = [
            folder for folder in os.listdir(self.train_dir) 
            if os.path.isdir(os.path.join(self.train_dir, folder))
        ]
        self.class_folders.sort()  # �ϰ����� ���� ����
        print(f"�� Ŭ���� ��: {len(self.class_folders)}")
        return self.class_folders
    
    def extract_image_paths(self, supported_formats: Tuple[str, ...] = ('.jpg',)) -> List[Dict]:
        """�� Ŭ���� �������� �̹��� ���� ��θ� ����"""
        if not self.class_folders:
            self.get_class_folders()
            
        self.image_data = []
        img_idx = 0
        
        for class_folder in self.class_folders:
            class_path = os.path.join(self.train_dir, class_folder)
            
            if not os.path.exists(class_path):
                print(f"���: {class_path} ��ΰ� �������� �ʽ��ϴ�.")
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(supported_formats):
                    # ID ����
                    img_id = f'TRAIN_{str(img_idx).zfill(5)}'
                    # �̹��� ���� ��� ����
                    img_path = os.path.join(class_path, img_file)
                    
                    # �̹��� ������ �߰�
                    self.image_data.append({
                        'ID': img_id,
                        'img_path': img_path,
                        'class_folder': class_folder
                    })
                    img_idx += 1
        
        print(f"�� �̹��� ��: {len(self.image_data)}")
        return self.image_data
    
    def create_one_hot_dataframe(self) -> pd.DataFrame:
        """��-�� ���ڵ��� ������������ ����"""
        if not self.image_data:
            self.extract_image_paths()
            
        # ������������ ����
        df = pd.DataFrame(self.image_data)
        
        # Ŭ������ �÷� �߰��ϰ� ��-�� ���ڵ�
        for class_name in self.class_folders:
            df[class_name] = 0.0
            # �̹����� Ŭ���� �������� ���� Ŭ������� ��ġ�ϸ� 1.0���� ����
            df.loc[df['class_folder'] == class_name, class_name] = 1.0
        
        # 'class_folder' �÷� ���� (���� CSV���� �ʿ� ����)
        df.drop('class_folder', axis=1, inplace=True)
        
        return df
    
    def save_to_csv(self, output_path: str) -> None:
        """�������������� CSV ���Ϸ� ����"""
        df = self.create_one_hot_dataframe()
        df.to_csv(output_path, index=False)
        print(f"CSV ������ �����Ǿ����ϴ�: {output_path}")
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Ŭ������ �̹��� ���� ���� ��ȯ"""
        if not self.image_data:
            self.extract_image_paths()
            
        distribution = {}
        for data in self.image_data:
            class_name = data['class_folder']
            distribution[class_name] = distribution.get(class_name, 0) + 1
            
        return distribution
