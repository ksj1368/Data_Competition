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
    """이미지 필터링을 담당하는 클래스"""
    
    def __init__(self, model_name: str = 'resnet50', max_workers: int = 8):
        self.model_name = model_name
        self.max_workers = max_workers
        self.model = None
        self.preprocess = None
        self.car_classes = [468, 475, 479, 555, 656, 569, 575, 609, 675, 705, 717, 734, 751, 779, 817, 864, 867, 874]
        self._setup_model()
    
    def _setup_model(self) -> None:
        """모델과 전처리 파이프라인 초기화"""
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_name}")
            
        self.model.eval()
        
        # 전처리 파이프라인 설정
        self.preprocess = transforms.Compose([
            transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"{self.model_name} 모델이 로드되었습니다.")
    
    def classify_single_image(self, img_path: str, threshold: float = 0.1) -> Tuple[int, float]:
        """단일 이미지 분류"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs[0], dim=0)
            
            # 자동차 클래스들의 확률 합계 계산
            car_prob = probs[self.car_classes].sum().item()
            
            # 임계값 기준으로 라벨 결정 (0: 자동차, 1: 자동차 아님)
            label = 0 if car_prob > threshold else 1
            
            return label, car_prob
            
        except Exception as e:
            print(f"이미지 처리 오류 {img_path}: {e}")
            return 1, 0.0  # 오류 발생 시 자동차 아님으로 분류
    
    def classify_images_parallel(self, img_paths: List[str], threshold: float = 0.1) -> List[Tuple[int, float]]:
        """병렬 처리로 여러 이미지 분류"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_path = {
                executor.submit(self.classify_single_image, img_path, threshold): img_path 
                for img_path in img_paths
            }
            
            # 결과 수집
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"병렬 처리 오류 {img_path}: {e}")
                    results.append((1, 0.0))
        
        return results
    
    def filter_dataframe(self, df: pd.DataFrame, threshold: float = 0.1, 
                        img_path_column: str = 'img_path') -> pd.DataFrame:
        """데이터프레임에 필터링 결과 추가"""
        if img_path_column not in df.columns:
            raise ValueError(f"컬럼 '{img_path_column}'이 데이터프레임에 없습니다.")
        
        print("이미지 필터링을 시작합니다...")
        img_paths = df[img_path_column].tolist()
        
        # 병렬 처리로 분류 실행
        results = self.classify_images_parallel(img_paths, threshold)
        
        # 결과를 데이터프레임에 추가
        df_copy = df.copy()
        df_copy['label'], df_copy['prob'] = zip(*results)
        
        # 통계 출력
        car_count = (df_copy['label'] == 0).sum()
        non_car_count = (df_copy['label'] == 1).sum()
        
        print(f"필터링 완료:")
        print(f"- 자동차 이미지: {car_count}개")
        print(f"- 자동차 아님: {non_car_count}개")
        print(f"- 전체 이미지: {len(df_copy)}개")
        
        return df_copy
    
    def update_car_classes(self, new_car_classes: List[int]) -> None:
        """자동차 클래스 목록 업데이트"""
        self.car_classes = new_car_classes
        print(f"자동차 클래스가 업데이트되었습니다: {len(self.car_classes)}개 클래스")
    
    def get_filter_statistics(self, df: pd.DataFrame) -> dict:
        """필터링 통계 반환"""
        if 'label' not in df.columns or 'prob' not in df.columns:
            raise ValueError("필터링이 적용되지 않은 데이터프레임입니다.")
        
        stats = {
            'total_images': len(df),
            'car_images': (df['label'] == 0).sum(),
            'non_car_images': (df['label'] == 1).sum(),
            'avg_car_probability': df[df['label'] == 0]['prob'].mean() if (df['label'] == 0).any() else 0,
            'avg_non_car_probability': df[df['label'] == 1]['prob'].mean() if (df['label'] == 1).any() else 0
        }
        
        return stats
