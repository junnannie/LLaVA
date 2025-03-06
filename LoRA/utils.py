from lora_layer import LoRALayer
from torch import nn


def apply_lora(model, target_modules, lora_rank, lora_alpha):
    """
    @param model: 原始模型
    @param target_modules: 需要替换的模块名称列表
    @param lora_rank: LoRA rank
    @param lora_alpha: LoRA alpha
    """

    # 遍历模型的顶层的子模块
    for name, module in model.named_children():
        # 1.vision_tower == SiglipVisionModel
        # 2.multi_modal_projector
        # 3.language_model == Qwen2ForCausalLM


        # 如果当前子模块不是叶子节点，递归遍历
        if len(list(module.children())) > 0:
            apply_lora(module, target_modules, lora_rank, lora_alpha)
        

        # 是线性层 且 名称在要被替换的 target_modules 中
        if isinstance(module, nn.Linear) and name in target_modules:

            # 替换线性层为LoRALayer
            new_layer = LoRALayer(module, lora_rank, lora_alpha)

            # name 的模块 替换成 new_layer
            setattr(model, name, new_layer)
    return model

def load_lora_weight(model, lora_weights=None):
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            if lora_weights is None:
                # 未提供权重时，将模型还原（lora 权重为 0）
                param.data.zero_()
            elif name in lora_weights:
                param.data.copy_(lora_weights[name])
            else:
                print(f"Warning: {name} not found in loaded lora_weights.")
