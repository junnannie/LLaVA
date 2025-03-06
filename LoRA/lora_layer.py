import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        dtype = original_layer.weight.dtype
        
        # 原始权重
        self.original_weight = original_layer.weight
        self.original_bias = original_layer.bias
        
        # 冻结原始参数
        self.original_weight.requires_grad_(False)
        if self.original_bias is not None:
            self.original_bias.requires_grad_(False)
        
        # LoRA参数
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype))
        
    def forward(self, x):
        # 原始层输出
        original_out = F.linear(x, self.original_weight, self.original_bias)
        
        # LoRA输出
        lora_out = torch.matmul(
            torch.matmul(x, self.lora_A), 
            self.lora_B
        ) * (self.alpha / self.rank)
        
        return original_out + lora_out
