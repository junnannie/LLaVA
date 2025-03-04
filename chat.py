import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def main(model_path, quantization = False, img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"):
    # model_path = "/mnt/workspace/llava-1.5-7b-hf"
    model_path = "/mnt/workspace/lora_llava_finetuned"

    # 配置量化参数
    bnb_config = BitsAndBytesConfig(
        # INT8 量化
        load_in_8bit=True,

        # 跳过视觉编码器量化
        llm_int8_skip_modules=['vision_tower', 'multi_modal_projector'],
        # 激活值异常阈值
        llm_int8_threshold=6.0,
    )

    if quantization:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,)
        model = model.to(0)

    model.eval()

    # 加载处理器参数说明：
    # 与模型同路径确保tokenizer和image_processor版本匹配
    processor = AutoProcessor.from_pretrained(model_path)


    img_url = img_url

    # stream=True: 流式传输避免大文件内存溢出
    response = requests.get(img_url, stream=True)
    raw_image = Image.open(response.raw)


    while True:
        user_input = input("🤗：")
        if user_input == "exit":
            break

        if user_input == "image":
            img_url = input("图片链接：")
            response = requests.get(img_url, stream=True)
            raw_image = Image.open(response.raw)
            continue

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
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        output_text = processor.decode(output[0], skip_special_tokens=True)
        # 只输出 ‘ASSISTANT’ 后面的文本
        print(f'🤖：{output_text.split("assistant")[-1].strip()}')
        # print(f'🤖：{output_text.split("ASSISTANT:")[-1].strip()}')

if __name__ == "__main__":

    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    
    # model_path = "/mnt/workspace/llava-1.5-7b-hf"
    qunatization = True

    model_path = "/mnt/workspace/lora_llava_finetuned"
    qunatization = False

    main(model_path, qunatization, img_url)