import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from pathlib import Path
import requests
from PIL import Image

from data.dataset import MyImageInstructionDataset
from utils import apply_lora
from lora_config import LoraConfig


MODEL_PATH = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"
LORA_WEIGHT = "/mnt/workspace/lora_llava_finetuned_manual/lora_weights.bin"

lora_config = LoraConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None


def main():
    global processor

    # 加载模型和处理器
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 加入 LoRA 结构
    model = apply_lora(model, lora_config.target_modules, lora_config.rank, lora_config.alpha)
    model = model.to(device)
    print('model:')
    print(model)

    # 加载 LoRA 权重
    lora_weights = torch.load(LORA_WEIGHT, map_location=device)

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            if name in lora_weights:
                param.data.copy_(lora_weights[name])
            else:
                print(f"Warning: {name} not found in loaded lora_weights.")

    model.eval()

    # 构建多模态对话模板
    # role: 用户提问角色标识
    # type=image: 图像输入的占位符标记
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "图片里是猫还是狗？"},
            {"type": "image"},
        ],
    }]

    # 将对话转换为模型输入格式
    # add_generation_prompt: 添加回复引导符（如"ASSISTANT:"）
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # 获取网络图片并加载
    # stream=True: 流式传输避免大文件内存占用
    image_response = requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        stream=True
    )
    raw_image = Image.open(image_response.raw)

    # 多模态输入预处理
    # images: 处理后的图像张量 (1, 3, 224, 224)
    # text: 格式化后的文本序列
    # return_tensors='pt': 返回PyTorch张量格式
    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt'
    ).to(0, torch.float16)  # 保持半精度并传至GPU

    # 执行生成推理
    # max_new_tokens: 控制生成文本最大长度
    # do_sample=False: 使用贪婪解码保证确定性输出
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    # 解码生成结果
    # skip_special_tokens=True: 过滤掉特殊终止符号
    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
