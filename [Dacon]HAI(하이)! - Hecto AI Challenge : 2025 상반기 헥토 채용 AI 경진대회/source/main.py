import pandas as pd
import torch
import os
from train import train_optimized_model
from predict import generate_optimized_submission
from utils import validate_paths
from data_preprocessor import DataPreprocessor
from image_filter import ImageFilter
from config import Config
import argparse

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='Car Image Classification Pipeline')
    parser.add_argument('--train-dir', type=str, help='Training data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--test-csv', type=str, help='Test CSV file path')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-filtering', action='store_true', 
                       help='Skip image filtering step')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    
    return parser.parse_args()

def main():
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 경로 설정
    paths = Config.get_paths(
        custom_train_dir=args.train_dir,
        custom_output_dir=args.output_dir
    )
    filter_config = Config.get_filter_config()
    
    # 경로 유효성 검사
    if not validate_paths(paths):
        print("경로 유효성 검사 실패")
        return False
    
    try:
        # 1. 데이터 전처리
        if not args.skip_preprocessing:
            print("\n=== 1. 데이터 전처리===")
            if not os.path.exists(paths['train_csv']):
                preprocessor = DataPreprocessor(paths['train_dir'])
                preprocessor.save_to_csv(paths['train_csv'])
                print(f"CSV 파일이 생성되었습니다: {paths['train_csv']}")
            else:
                print(f"기존 CSV 파일을 사용합니다: {paths['train_csv']}")

        # 2. 이미지 필터링
        filtered_csv_path = paths['filtered_csv']
        if not args.skip_filtering:
            print("\n=== 2. 이미지 필터링 ===")
            if not os.path.exists(filtered_csv_path):
                filter_model = ImageFilter(
                    model_name=filter_config['model_name'],
                    max_workers=filter_config['max_workers']
                )
                df = pd.read_csv(paths['train_csv'])
                
                filtered_df = filter_model.filter_dataframe(
                    df, 
                    threshold=filter_config['threshold']
                )
                
                # 필터링된 결과 저장
                filtered_df.to_csv(filtered_csv_path, index=False)
                print(f"필터링된 CSV 파일이 저장되었습니다: {filtered_csv_path}")
                
                # 필터링 통계 출력
                stats = filter_model.get_filter_statistics(filtered_df)
                print(f"필터링 통계: {stats}")
            else:
                print(f"기존 필터링된 CSV 파일을 사용합니다: {filtered_csv_path}")
        else:
            # 필터링을 건너뛸 경우 원본 CSV 사용
            filtered_csv_path = paths['train_csv']

        # GPU 메모리 상태 모니터링
        print(f"\n=== GPU 메모리 상태 ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB")

        # 3. 모델 훈련
        print("\n=== 3. 모델 훈련 ===")
        model = train_optimized_model(
            train_csv=filtered_csv_path,
            val_split=0.1,
            epochs=args.epochs,
            accumulation_steps=8,
            resume_path=args.resume
        )

        # 4. 예측 및 제출 파일 생성
        model_path = os.path.join(paths['output_dir'], 'best_convnext_384_model.pth')
        if os.path.exists(model_path):
            print("\n=== 4. 예측 및 제출 파일 생성 ===")
            test_csv_path = args.test_csv if args.test_csv else './hecto-dataset/test.csv'
            
            if os.path.exists(test_csv_path):
                generate_optimized_submission(model_path, test_csv_path)
                print("제출 파일 생성이 완료되었습니다.")
            else:
                print(f"테스트 CSV 파일을 찾을 수 없습니다: {test_csv_path}")
        else:
            print(f"훈련된 모델을 찾을 수 없습니다: {model_path}")
            
        return True
        
    except Exception as e:
        print(f"실행 중 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """실행 예시: python main.py --train-dir ./hecto-dataset/train --output-dir /result --epochs 50"""
    success = main()
    if success:
        print("\n프로그램이 성공적으로 완료되었습니다.")
    else:
        print("\n프로그램 실행 중 오류가 발생했습니다.")
        exit(1)