

Step1：创建 conda 环境

```bash
conda create -n llava python=3.10
conda activate llava
```



Step2：配置 LLaVA

```bash
# 安装依赖
pip install transformers torch torchvision torchaudio peft bitsandbytes openai tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install 'accelerate>=0.26.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
```



Step3：git clone

```bash
git clone https://github.com/niejnan/LLM.git
```



Step4：下载模型权重

```bash
# 安装 git-lfs 
apt install git-lfs
git lfs install

# 下载权重，需要等待一段时间
git clone https://www.modelscope.cn/llava-hf/llava-interleave-qwen-0.5b-hf.git

# LLaVA 7B
git clone https://www.modelscope.cn/llava-hf/llava-1.5-7b-hf.git
```



Step5：构造数据（可选）

`data/make_data.py` 第 X 行，修改 API Key，推荐用 DeepSeek-V3，在 00:30-08:30 调用，主打的就是一个便宜

```bash
api_key = '你的 API-KEY'
```

根据想实现的风格，修改 `prompt`，例如：

> 请使用广东话风格来回答，语气应带有懒散、悠闲的感觉，尽量使用粤语中的地道词汇和语调。回答要充满生活气息，带点调侃和自信，语气应显得轻松自在且不急不躁。
>
> 比如：“咩啊，唔使咁紧张啦，啲嘢都可以慢慢嚟，唔使急，饮杯茶啦，唔好咁攰。”
>
> 让回答充满广东人特有的轻松和幽默感，表达不拘小节但又十分温暖的态度。



SA1B:https://www.modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption/summary

