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
    """����� �μ� �Ľ�"""
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
    # ����� �μ� �Ľ�
    args = parse_arguments()
    
    # ��� ����
    paths = Config.get_paths(
        custom_train_dir=args.train_dir,
        custom_output_dir=args.output_dir
    )
    filter_config = Config.get_filter_config()
    
    # ��� ��ȿ�� �˻�
    if not validate_paths(paths):
        print("��� ��ȿ�� �˻� ����")
        return False
    
    try:
        # 1. ������ ��ó��
        if not args.skip_preprocessing:
            print("\n=== 1. ������ ��ó��===")
            if not os.path.exists(paths['train_csv']):
                preprocessor = DataPreprocessor(paths['train_dir'])
                preprocessor.save_to_csv(paths['train_csv'])
                print(f"CSV ������ �����Ǿ����ϴ�: {paths['train_csv']}")
            else:
                print(f"���� CSV ������ ����մϴ�: {paths['train_csv']}")

        # 2. �̹��� ���͸�
        filtered_csv_path = paths['filtered_csv']
        if not args.skip_filtering:
            print("\n=== 2. �̹��� ���͸� ===")
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
                
                # ���͸��� ��� ����
                filtered_df.to_csv(filtered_csv_path, index=False)
                print(f"���͸��� CSV ������ ����Ǿ����ϴ�: {filtered_csv_path}")
                
                # ���͸� ��� ���
                stats = filter_model.get_filter_statistics(filtered_df)
                print(f"���͸� ���: {stats}")
            else:
                print(f"���� ���͸��� CSV ������ ����մϴ�: {filtered_csv_path}")
        else:
            # ���͸��� �ǳʶ� ��� ���� CSV ���
            filtered_csv_path = paths['train_csv']

        # GPU �޸� ���� ����͸�
        print(f"\n=== GPU �޸� ���� ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB")

        # 3. �� �Ʒ�
        print("\n=== 3. �� �Ʒ� ===")
        model = train_optimized_model(
            train_csv=filtered_csv_path,
            val_split=0.1,
            epochs=args.epochs,
            accumulation_steps=8,
            resume_path=args.resume
        )

        # 4. ���� �� ���� ���� ����
        model_path = os.path.join(paths['output_dir'], 'best_convnext_384_model.pth')
        if os.path.exists(model_path):
            print("\n=== 4. ���� �� ���� ���� ���� ===")
            test_csv_path = args.test_csv if args.test_csv else './hecto-dataset/test.csv'
            
            if os.path.exists(test_csv_path):
                generate_optimized_submission(model_path, test_csv_path)
                print("���� ���� ������ �Ϸ�Ǿ����ϴ�.")
            else:
                print(f"�׽�Ʈ CSV ������ ã�� �� �����ϴ�: {test_csv_path}")
        else:
            print(f"�Ʒõ� ���� ã�� �� �����ϴ�: {model_path}")
            
        return True
        
    except Exception as e:
        print(f"���� �� ������ �߻��߽��ϴ�: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """���� ����: python main.py --train-dir ./hecto-dataset/train --output-dir /result --epochs 50"""
    success = main()
    if success:
        print("\n���α׷��� ���������� �Ϸ�Ǿ����ϴ�.")
    else:
        print("\n���α׷� ���� �� ������ �߻��߽��ϴ�.")
        exit(1)