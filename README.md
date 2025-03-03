# 多模态风格化问答大模型

该项目是一个多模态风格化问答大模型，结合了视觉和大语言模型，旨在实现基于图像和文本输入的智能问答和风格化输出。

1. **轻量化的 LLaVA 架构**：该模型基于 Qwen2-0.5B 语言模型和自监督预训练的 SigLIP 视觉模型，通过线性投影构造了一个参数量仅为 0.8B 的轻量化多模态模型。

2. **风格化指令数据集**：采用 [SA1B-长文本图文描述](https://www.modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption/summary) 数据集，筛选出 1000 组高质量的图文对，并利用 DeepSeek-V3 实现指令风格迁移，生成广东俚语、暴躁老哥等风格的指令问答对，从而构建了适用于视觉问答和场景解析任务的风格化训练集。

3. **高效参数微调**：通过自主实现 LoRA 微调，仅训练模型约 0.35% 的参数（约 300 万参数）。通过对比试验验证，LoRA 微调与全量微调在效果上几乎无差异，确保了高效的训练性能。



## 安装

### Step1：创建 conda 环境

```bash
conda create -n llava python=3.10
conda activate llava
```



### Step2：配置 LLaVA

```bash
# 安装依赖
pip install transformers torch torchvision torchaudio peft bitsandbytes openai tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install 'accelerate>=0.26.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
```



### Step3：git clone

```bash
git clone https://github.com/niejnan/LLM.git
```



### Step4：下载模型权重

```bash
# 安装 git-lfs 
apt install git-lfs
git lfs install

# 下载权重，需要等待一段时间
git clone https://www.modelscope.cn/llava-hf/llava-interleave-qwen-0.5b-hf.git

# LLaVA 7B
git clone https://www.modelscope.cn/llava-hf/llava-1.5-7b-hf.git
```



### Step5：构造数据（可选）

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

