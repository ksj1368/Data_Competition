import torch
import gc
import os
from typing import Dict

def validate_paths(paths: Dict[str, str]) -> bool:
    """��� ��ȿ�� �˻�"""
    if not os.path.exists(paths['train_dir']):
        print(f"����: �н� ������ ���丮�� �������� �ʽ��ϴ�: {paths['train_dir']}")
        return False
    
    if not os.path.exists(paths['output_dir']):
        os.makedirs(paths['output_dir'], exist_ok=True)
        print(f"��� ���丮�� �����߽��ϴ�: {paths['output_dir']}")
    
    return True

def cleanup_memory():
    """GPU �޸� ����"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def setup_multi_gpu():
    """��Ƽ GPU ȯ�� ���� �� ����ȭ"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    
    # CUDA �޸� ���� ����ȭ
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print(f"Number of available GPUs: {gpu_count}")
    if gpu_count > 1:
        print("Run in multi-GPU mode.")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    return device, gpu_count
