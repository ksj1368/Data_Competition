import torch
import torch.nn as nn
from torchvision import models
from torch.utils.checkpoint import checkpoint

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # ConvNeXt-Base backbone(Pretrained ImageNet Weights)
        self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        
        # ConvNeXt-Base input
        in_features = self.backbone.classifier[2].in_features
        
        # Stanford Cars에 최적화된 분류기 설계
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(0.2),
            nn.Linear(in_features, 1024),
            nn.GELU(), 
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 새로운 분류기로 교체
        self.backbone.classifier = self.classifier

    def forward(self, x):
        if self.use_checkpoint and self.training:
            # Gradient Checkpointing 적용
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        return self.backbone(x)
