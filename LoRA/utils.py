from lora_layer import LoRALayer
from torch import nn


def apply_lora(model, target_modules, lora_rank, lora_alpha):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_lora(module, target_modules, lora_rank, lora_alpha)
            
        if isinstance(module, nn.Linear) and name in target_modules:
            # 替换线性层为LoRALayer
            new_layer = LoRALayer(module, lora_rank, lora_alpha)
            setattr(model, name, new_layer)
    return model
