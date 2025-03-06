import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from pathlib import Path

from data.dataset import MyImageInstructionDataset
from utils import apply_lora
from lora_config import LoraConfig

# ----------------- 超参数配置 -----------------
MODEL_PATH = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"
DATA_PATH = "./data/writer.json"
OUTPUT_DIR = "/mnt/workspace/lora_llava_finetuned_manual"
MODEL_NAME = "writer"

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
lora_config = LoraConfig()

# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None


def collate_fn(batch):
    """
    自定义 batch 处理函数：
    1. 对于每个样本，根据 question 与 answer 构造对话 prompt；
    2. 收集图像和文本后，调用 processor 同时编码图像和文本；
    3. 将 input_ids 复制到 labels，并对用户输入部分进行屏蔽
    """
    images = []
    texts = []
    for item in batch:
        # 构造对话：用户包含问题与图像占位符，助手为回答
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item["question"]},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["answer"]}
                ]
            }
        ]
        # 根据对话构造 prompt，add_generation_prompt 根据需求选择是否自动添加生成提示
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        texts.append(prompt)
        images.append(item["image"])
    
    # 通过 processor 同时编码图像与文本，返回字典包含 input_ids、attention_mask、pixel_values 等字段
    model_inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)

    # 将 input_ids 复制为 labels
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    assistant_token_id = 77091
    for idx, input_ids in enumerate(model_inputs["input_ids"]):
        # 找到所有assistant标记的位置
        assistant_positions = (input_ids == assistant_token_id).nonzero()
        
        if len(assistant_positions) > 0:
            # 取第一个出现的位置
            start_pos = assistant_positions[0].item() + 1  # +1 跳过角色标记本身
            model_inputs["labels"][idx, :start_pos] = -100
        else:
            # 处理没有找到的情况（例如屏蔽全部）
            model_inputs["labels"][idx, :] = -100

    return model_inputs

def main():
    global processor

    # 1. 加载模型和处理器
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print('model:')
    print(model)

    # 2. 手动应用LoRA
    model = apply_lora(model, lora_config.target_modules, lora_config.rank, lora_config.alpha)
    model = model.to(device)
    print('model:')
    print(model)
    
    # 3. 设置可训练参数
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"可训练参数数量: {len(trainable_params)}")
    
    # 4. 加载数据集
    dataset = MyImageInstructionDataset(DATA_PATH, transform=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 5. 优化器设置
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    # 6. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in epoch_iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_iterator.set_postfix(loss=loss.item())
    
    # 7. 保存 LoRA 权重
    lora_weights = {
        name: param.data for name, param in model.named_parameters()
        if 'lora_A' in name or 'lora_B' in name
    }
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    torch.save(lora_weights, f"{OUTPUT_DIR}/{MODEL_NAME}.bin")

if __name__ == "__main__":
    main()
