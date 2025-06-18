import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import setup_multi_gpu, cleanup_memory
from dataset import CarDataset, get_optimized_batch_size
from model import ConvNeXtModel

def memory_efficient_predict(model, test_loader, device):
    """메모리 효율적인 예측 함수"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc="prediction in progress"):
            images = images.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                batch_preds = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(batch_preds)
            
            # 메모리 정리
            del outputs
            if len(all_preds) % 10 == 0:
                cleanup_memory()
    
    return np.concatenate(all_preds)

def load_optimized_checkpoint(checkpoint_path, device):
    """ConvNeXt 최적화된 체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ConvNeXtModel(
        num_classes=checkpoint['model_config']['num_classes'],
        use_checkpoint=False,  # 추론 시에는 체크포인팅 비활성화
    )
    
    current_gpu_count = torch.cuda.device_count()
    
    if current_gpu_count > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # state_dict 로드
    state_dict = checkpoint['state_dict']
    if current_gpu_count > 1 and not any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {f'module.{key}': value for key, value in state_dict.items()}
    elif current_gpu_count == 1 and any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"Model load completed - Backbone: {checkpoint['model_config'].get('backbone', 'unknown')}")
    return model

def generate_optimized_submission(model_path, test_csv):
    """최적화된 제출 파일 생성"""
    device, gpu_count = setup_multi_gpu()
    
    test_df = pd.read_csv(test_csv)
    test_df["img_path"] = test_df["img_path"].str.replace(
        "./test", "./hecto-dataset/test", regex=False
    )
    
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CarDataset(test_df, transform=test_transform, is_test=True)
    batch_size = get_optimized_batch_size(gpu_count, base_batch_size=32)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print("load optimized checkpoint...")
    model = load_optimized_checkpoint(model_path, device)
    
    print("memory efficient predicting...")
    predictions = memory_efficient_predict(model, test_loader, device)
    
    # 개선된 확률 보정
    def advanced_calibrate_probs(probs, temperature=0.95, sharpening=1.2):
        # 온도 스케일링
        probs = probs ** (1/temperature)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        # 샤프닝
        probs = probs ** sharpening
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    calibrated_probs = advanced_calibrate_probs(predictions)
    
    # 제출 파일 생성
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    class_names = checkpoint['class_names']
    
    submission_df = pd.DataFrame(calibrated_probs, columns=class_names)
    submission_df.insert(0, 'ID', test_df['ID'])
    sub_csv_name = 'convnext_submission.csv'
    submission_df.to_csv(sub_csv_name, index=False)
    
    print(f"The optimized submission file has been saved as {sub_csv_name}.")
    print(f"Model performance - Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    # 메모리 정리
    cleanup_memory()
    return submission_df
