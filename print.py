import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from data.dataset import MyImageInstructionDataset


MODEL_PATH = "./llava-interleave-qwen-0.5b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

all_params = []
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Shape: {param.shape}")
    # 将参数从 GPU 移回 CPU 并转换为 numpy 数组，展平成一维向量
    all_params.append(param.detach().cpu().numpy().flatten())

# 将所有参数合并为一个一维数组
all_params = np.concatenate(all_params)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(all_params, bins=100, color='skyblue', edgecolor='black')
plt.title("Model Parameter Distribution")
plt.xlabel("Parameter Value")
plt.ylabel("Frequency")
plt.show()