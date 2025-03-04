import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

# 原始模型和LoRA适配器的存储路径
model_path = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"
lora_path = "/mnt/workspace/lora_llava_finetuned"

# 加载基础视觉语言模型
# torch_dtype: 半精度浮点节省显存
# low_cpu_mem_usage: 优化大模型加载的内存占用
# to(0): 部署到第一个GPU设备
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

# 初始化多模态处理器（处理文本+图像）
processor = AutoProcessor.from_pretrained(model_path)

# 加载LoRA适配器参数
# 通过PEFT库将微调后的低秩矩阵加载到基础模型
model = PeftModel.from_pretrained(model, lora_path)

# (可选)合并LoRA权重到基础模型
# 提升推理速度但会失去适配器可调性
# model = model.merge_and_unload()

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
