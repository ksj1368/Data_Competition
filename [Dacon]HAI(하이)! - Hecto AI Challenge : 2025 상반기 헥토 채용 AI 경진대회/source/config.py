import os
from typing import Dict, Any

class Config:
    """���� ���� Ŭ����"""
    
    # �⺻ ��� ����
    DEFAULT_TRAIN_DIR = './hecto-dataset/train'
    DEFAULT_OUTPUT_DIR = './result'
    
    # �̹��� ���͸� ����
    DEFAULT_FILTER_THRESHOLD = 0.1
    DEFAULT_MAX_WORKERS = 8
    DEFAULT_MODEL_NAME = 'resnet50'
    
    # �����ϴ� �̹��� ����
    SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')
    
    # ImageNet �ڵ��� ���� Ŭ���� ID
    CAR_CLASSES = [468, 475, 479, 555, 656, 569, 575, 609, 675, 705, 717, 734, 751, 779, 817, 864, 867, 874]
    
    @classmethod
    def get_paths(cls, custom_train_dir: str = None, custom_output_dir: str = None) -> Dict[str, str]:
        """��� ���� ��ȯ"""
        return {
            'train_dir': custom_train_dir or cls.DEFAULT_TRAIN_DIR,
            'output_dir': custom_output_dir or cls.DEFAULT_OUTPUT_DIR,
            'train_csv': os.path.join(custom_output_dir or cls.DEFAULT_OUTPUT_DIR, 'train.csv'),
            'filtered_csv': os.path.join(custom_output_dir or cls.DEFAULT_OUTPUT_DIR, 'train_filter.csv')
        }
    
    @classmethod
    def get_filter_config(cls) -> Dict[str, Any]:
        """���͸� ���� ��ȯ"""
        return {
            'threshold': cls.DEFAULT_FILTER_THRESHOLD,
            'max_workers': cls.DEFAULT_MAX_WORKERS,
            'model_name': cls.DEFAULT_MODEL_NAME,
            'car_classes': cls.CAR_CLASSES
        }
