import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 本地模型路径（需替换为实际路径）
model_path = "/mnt/workspace/llava-interleave-qwen-0.5b-hf"

# 加载模型参数说明：
# pretrained_model_name_or_path: 模型本地路径
# torch_dtype=torch.float16: 半精度浮点节省50%显存
# low_cpu_mem_usage=True: 分片加载减少内存峰值
# .to('cuda:0'): 部署到第一个GPU
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to('cuda:0')

# 加载处理器参数说明：
# 与模型同路径确保tokenizer和image_processor版本匹配
processor = AutoProcessor.from_pretrained(model_path)

# 对话模板参数说明：
# role: 用户角色提问
# content: 多模态输入（text类型为问题，image类型为占位符）
conversation = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "图片里是什么？"},
        {"type": "image"},
    ],
}]

# 生成prompt参数说明：
# add_generation_prompt=True: 自动添加"\nASSISTANT:"引导生成
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 下载图片参数说明：
# stream=True: 流式传输避免大文件内存溢出
response = requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True)
raw_image = Image.open(response.raw)

# 预处理参数说明：
# images: 输入PIL图像对象
# text: 格式化后的prompt文本
# return_tensors='pt': 返回PyTorch张量
# .to(0, torch.float16): 数据传至GPU0+半精度
inputs = processor(
    images=raw_image,
    text=prompt,
    return_tensors='pt'
).to(0, torch.float16)

# 生成参数说明：
# max_new_tokens=200: 限制生成最大token数量
output = model.generate(
    **inputs,
    max_new_tokens=200,
)

# 解码参数说明：
# skip_special_tokens=False: 保留</s>等特殊标记
print(processor.decode(output[0], skip_special_tokens=False))
