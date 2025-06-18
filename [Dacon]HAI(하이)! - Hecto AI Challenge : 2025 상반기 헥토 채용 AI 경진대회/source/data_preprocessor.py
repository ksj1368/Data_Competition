import numpy as np
import pandas as pd
import os
import warnings
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """데이터 전처리를 담당하는 클래스"""
    
    def __init__(self, train_dir: str):
        self.train_dir = train_dir
        self.class_folders = []
        self.image_data = []
        
    def get_class_folders(self) -> List[str]:
        """클래스 폴더명을 가져오고 정렬"""
        self.class_folders = [
            folder for folder in os.listdir(self.train_dir) 
            if os.path.isdir(os.path.join(self.train_dir, folder))
        ]
        self.class_folders.sort()  # 일관성을 위해 정렬
        print(f"총 클래스 수: {len(self.class_folders)}")
        return self.class_folders
    
    def extract_image_paths(self, supported_formats: Tuple[str, ...] = ('.jpg',)) -> List[Dict]:
        """각 클래스 폴더에서 이미지 파일 경로를 추출"""
        if not self.class_folders:
            self.get_class_folders()
            
        self.image_data = []
        img_idx = 0
        
        for class_folder in self.class_folders:
            class_path = os.path.join(self.train_dir, class_folder)
            
            if not os.path.exists(class_path):
                print(f"경고: {class_path} 경로가 존재하지 않습니다.")
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(supported_formats):
                    # ID 생성
                    img_id = f'TRAIN_{str(img_idx).zfill(5)}'
                    # 이미지 파일 경로 생성
                    img_path = os.path.join(class_path, img_file)
                    
                    # 이미지 데이터 추가
                    self.image_data.append({
                        'ID': img_id,
                        'img_path': img_path,
                        'class_folder': class_folder
                    })
                    img_idx += 1
        
        print(f"총 이미지 수: {len(self.image_data)}")
        return self.image_data
    
    def create_one_hot_dataframe(self) -> pd.DataFrame:
        """원-핫 인코딩된 데이터프레임 생성"""
        if not self.image_data:
            self.extract_image_paths()
            
        # 데이터프레임 생성
        df = pd.DataFrame(self.image_data)
        
        # 클래스별 컬럼 추가하고 원-핫 인코딩
        for class_name in self.class_folders:
            df[class_name] = 0.0
            # 이미지의 클래스 폴더명이 현재 클래스명과 일치하면 1.0으로 설정
            df.loc[df['class_folder'] == class_name, class_name] = 1.0
        
        # 'class_folder' 컬럼 제거 (최종 CSV에는 필요 없음)
        df.drop('class_folder', axis=1, inplace=True)
        
        return df
    
    def save_to_csv(self, output_path: str) -> None:
        """데이터프레임을 CSV 파일로 저장"""
        df = self.create_one_hot_dataframe()
        df.to_csv(output_path, index=False)
        print(f"CSV 파일이 생성되었습니다: {output_path}")
        
    def get_class_distribution(self) -> Dict[str, int]:
        """클래스별 이미지 개수 분포 반환"""
        if not self.image_data:
            self.extract_image_paths()
            
        distribution = {}
        for data in self.image_data:
            class_name = data['class_folder']
            distribution[class_name] = distribution.get(class_name, 0) + 1
            
        return distribution
