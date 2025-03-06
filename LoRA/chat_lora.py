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
from utils import apply_lora, load_lora_weight
from lora_config import LoraConfig


# 保存多种 lora，用于动态加载
lora_dict = {
    "base": {
        "path": None,
        "weight": None,
    },
    "anger": {
        "path": "/mnt/workspace/lora_llava_finetuned_manual/anger.bin",
        "weight": None,
    },
    "writer": {
        "path": "/mnt/workspace/lora_llava_finetuned_manual/writer.bin",
        "weight": None,
    }
}

# base model 路径
MODEL_PATH = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"

# lora 配置，必须与 lora 训练时一致
lora_config = LoraConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 加载模型和处理器
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.generation_config.pad_token_id = 151645
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 加入 LoRA 结构
    model = apply_lora(model, lora_config.target_modules, lora_config.rank, lora_config.alpha)
    model = model.to(device)
    print('model:')
    print(model)

    # 加载 LoRA 权重到内存
    for each in lora_dict.values():
        if each["path"]:
            each["weight"] = torch.load(each["path"])

    model.eval()

    # 下载图片参数说明：
    # stream=True: 流式传输避免大文件内存溢出
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(img_url, stream=True)
    raw_image = Image.open(response.raw)


    while True:
        user_input = input("🤗：")
        if user_input == "exit":
            break

        if user_input == "image":
            img_url = input("图片链接：")

            # 尝试获取图片
            try:
                response = requests.get(img_url, stream=True)
                raw_image = Image.open(response.raw)
                print("图片加载成功")
                continue
            except Exception as e:
                print(f"图片加载失败：{e}")
                continue

        if user_input == "lora":
            # 加载指定的 lora 权重
            lora_name = input("lora 名称：")
            if lora_name not in lora_dict:
                print(f"名为 {lora_name} 的 lora 权重不存在")
                continue

            load_lora_weight(model, lora_dict[lora_name]["weight"])
            continue

        # 构造模型输入
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {"type": "image"},
            ],
        }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            images=raw_image,
            text=prompt,
            return_tensors='pt'
        ).to(0)

        # 模型推理
        output = model.generate(
            **inputs,
            max_new_tokens=200,
        )

        # 将输出的 token 解码为文本
        output_text = processor.decode(output[0], skip_special_tokens=True)

        # 只输出 ‘ASSISTANT’ 后面的文本
        print(f'🤖：{output_text.split("assistant")[-1].strip()}')


if __name__ == "__main__":
    main()
