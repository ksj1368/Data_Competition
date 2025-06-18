import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sam import SAM

from utils import setup_multi_gpu, cleanup_memory
from dataset import CarDataset, get_train_transform, get_val_transform, get_optimized_batch_size
from model import ConvNeXtModel

def validate_with_amp(model, val_loader, device, criterion):
    """AMP를 적용한 검증 함수"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader.dataset)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_optimized_model(train_csv, val_split=0.1, epochs=25, accumulation_steps=4, resume_path=None):
    """ConvNeXt 기반 메모리 최적화된 모델 학습"""
    # 시드 설정
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # GPU 환경 설정
    device, gpu_count = setup_multi_gpu()
    
    # 데이터 로드 및 전처리
    if train_csv == './result/train_filter.csv':
        df = pd.read_csv(train_csv, index_col=0)
        df = df[df['label'] == 0]
        df = df.drop(['label', 'prob'], axis=1)
    else:
        df = pd.read_csv(train_csv)
    
    class_names = df.columns[2:].tolist()
    labels = df.iloc[:, 2:].values.argmax(axis=1)
    
    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=labels, 
        random_state=seed, shuffle=True
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"Train data: {len(train_df)}, Validation data: {len(val_df)}")
    
    
    batch_size = get_optimized_batch_size(gpu_count, base_batch_size=32)
    num_workers = min(8, os.cpu_count())
    
    train_dataset = CarDataset(train_df, transform=get_train_transform())
    val_dataset = CarDataset(val_df, transform=get_val_transform())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True
    )
    
    # ConvNeXt-Base 모델 생성
    model = ConvNeXtModel(
        num_classes=len(class_names), 
        use_checkpoint=True
    )
    
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel applied. GPUs used: {list(range(gpu_count))}")
    
    model = model.to(device)
    
    base_optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-5 * gpu_count,
        weight_decay=0.05,
        eps=1e-8
    )
    
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=0.05
    )
    
    scheduler = CosineAnnealingWarmRestarts(base_optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AMP 스케일러 초기화
    scaler = GradScaler()
    
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7
    
    # 체크포인트 로드
    if resume_path and os.path.exists(resume_path):
        print(f"Load the model from checkpoint {resume_path}.")
        checkpoint_data = torch.load(resume_path, map_location=device)
        
        state_dict = checkpoint_data['state_dict']
        if gpu_count > 1 and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif gpu_count == 1 and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        
        if 'optimizer' in checkpoint_data:
            base_optimizer.load_state_dict(checkpoint_data['optimizer'])
        if 'scheduler' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler'])
        if 'epoch' in checkpoint_data:
            start_epoch = checkpoint_data['epoch'] + 1
            epochs += start_epoch + 1
        if 'val_acc' in checkpoint_data:
            best_val_acc = checkpoint_data['val_acc']
        if 'val_loss' in checkpoint_data:
            best_val_loss = checkpoint_data['val_loss']
    
    print("Start model training")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        cleanup_memory()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        # 학습
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Closure 함수 정의 (SAM의 두 번째 forward-backward pass용)
            def closure():
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels) / accumulation_steps
                scaler.scale(loss).backward()
                return loss
            
            # 첫 번째 forward-backward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer, closure)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                progress_bar.set_postfix({
                    'loss': loss.item() * accumulation_steps,
                    'acc': 100 * correct / total,
                    'mem_alloc': f'{memory_allocated:.2f}GB',
                    'mem_res': f'{memory_reserved:.2f}GB'
                })
        
        # 검증
        val_loss, val_acc = validate_with_amp(model, val_loader, device, criterion)
        scheduler.step()
        
        train_acc = 100 * correct / total
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 최고 성능 모델 저장
        if best_val_loss > val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            model_to_save = model.module if gpu_count > 1 else model
            torch.save({
                'state_dict': model_to_save.state_dict(),
                'class_names': class_names,
                'model_config': {
                    'num_classes': len(class_names),
                    'backbone': 'convnext_base',
                },
                'gpu_count': gpu_count,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'optimizer': base_optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, './result/best_convnext_384_model.pth')
            print(f'New top performance model saved. Val Loss: {val_loss:.2f}, Val acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        print('-' * 60)
    
    print(f"Model training completed. best val loss: {best_val_loss:.2f}, best val acc: {best_val_acc:.2f}%")
    cleanup_memory()
    return model
