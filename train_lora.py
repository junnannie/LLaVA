import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from data.dataset import MyImageInstructionDataset

# ----------------- 超参数配置 -----------------
MODEL_PATH = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"
DATA_PATH = "./data/data.json"
OUTPUT_DIR = "/mnt/workspace/lora_llava_finetuned"

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5

# ------------------------------------------------
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局声明 processor
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
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 2. 配置 LoRA 参数并应用 LoRA（target_modules 需要根据模型实际结构确定）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 检查只有 LoRA 部分参数是可训练的
    print(model)

    # 3. 加载数据集，并构造 DataLoader
    dataset = MyImageInstructionDataset(DATA_PATH, transform=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 4. 定义优化器，AdamW 把权重衰退
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        
        for step, batch in enumerate(epoch_iterator):
            # batch 是一个字典，包含 input_ids、attention_mask、pixel_values 和 labels
            # input_ids: 输入的 token 的 id，尺寸为 [batch_size, sequence_length]，下同
            # attention_mask: 表示输入 token 的 mask
            # labels: 目标的 token 的 id
            # pixel_values: 输入的图像的像素值，尺寸为 [batch_size, num_channels, height, width]

            # 将 batch 中的所有张量移到 device 上
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # 前向传播：输入包括 input_ids、attention_mask、pixel_values 和 labels
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # 使用 tqdm 显示当前 loss
            epoch_iterator.set_postfix(loss=loss.item())
            
    # 5. 训练结束后保存模型与处理器
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()