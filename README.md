# 多模态风格化问答大模型

该项目是针对视觉—语言任务的多模态大模型，结合了视觉和语言信息以进行推理，实现了支持图像和文本输入可定制风格化输出的智能问答助手。

1. **结合视觉和语言信息**：通过对视觉和语言的共同建模，使得大语言模型能够“理解”图像内容，生成相关的文字描述或是回答问题，实现在多模态任务中进行联合理解与推理。
2. **轻量化架构**：基于轻量的语言模型 Qwen2-0.5B 和视觉模型 SigLIP，通过自监督预训练方法进行多模态任务的训练，有效地降低了模型的参数量，并且具有更高的推理效率，更适合实际应用。
3. **风格化指令数据集**：利用了SA1B（长文本图文描述）数据集，结合 DeepSeek-V3 进行风格迁移，生成了如广东俚语、暴躁老哥、诚恳推销等多种风格化的指令问答对，能够在不同的上下文中生成符合情境的文本。
4. **高效参数微调**：自主实现 LoRA 微调，仅更新模型的 0.35% 的参数，大幅度减少计算开销，同时保持接近全量微调的效果，在任务性能和计算效率上达到了一个良好的平衡。



## 实际效果展示

- 🤗：表示用户输入

- 🤖：表示模型输出



#### [测试图片](https://modelscope.cn-beijing.oss.aliyuncs.com/open_data/sa-1b-cot-qwen/sa_5581992.jpg)：

![test1](images/test1.jpg)



#### 风格1：暴躁回答版(语气带有一点着急)

![2](images/2.png)



#### 风格2：广东本地版(使用粤语中的地道词汇和语调)

![3](images/3.png)



#### 风格3：推销员版

![4](images/4.png)





## 细节

#### 1.为什么要 LoRA 微调？

以 LLaVA-7B 为例，使用 `Float32`，模型参数 7B $\times$ 4 = 28GB，梯度 7B $\times$ 4 = 28GB， Adam 存储动量 7B $\times$ 8 = 56GB，这里就至少占用了 112G 显存了。所以要么量化到 `INT8`，要么就做 LoRA 微调。



#### 2. LoRA 微调下调整了多少参数量？

LoRA 中设置为：$r=8$，$\alpha=16$

```python
# Qwen-0.5B
trainable params: 3,489,792 || all params: 867,521,056 || trainable%: 0.4023
```



```python
# LLaVA-7B
trainable params: 9,961,472 || all params: 7,073,388,544 || trainable%: 0.1408
```



#### 3. 代码中奇怪的 `assistant_token_id = 77091` 是怎么回事？

```python
    assistant_token_id = 77091
    for idx, input_ids in enumerate(model_inputs["input_ids"]):
        # 找到所有assistant标记的位置
        assistant_positions = (input_ids == assistant_token_id).nonzero()
        # 取第一个出现的位置
        start_pos = assistant_positions[0].item() + 1
        model_inputs["labels"][idx, :start_pos] = -100
```

Qwen2的 `assistant` 的 token 的索引就是77091，这一段是为了找到模型回答内容中 assistant 的位置，只对 assistant 后面的回答做监督。



#### 4. 多少个 Decoder？

答：以 Qwen-0.5B 为例，千问模型中有24个 Decoder，26个 SiglipEncoder



## 安装

#### Step1：创建 conda 环境

```bash
conda create -n llava python=3.10
conda activate llava
```



#### Step2：配置 LLaVA

```bash
pip install transformers torch torchvision torchaudio peft bitsandbytes openai tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install 'accelerate>=0.26.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

git clone https://github.com/niejnan/LLaVA.git
```



#### Step3：下载模型权重

```bash
apt install git-lfs
git lfs install

# Qwen-0.5B
git clone https://www.modelscope.cn/llava-hf/llava-interleave-qwen-0.5b-hf.git

# LLaVA 7B
git clone https://www.modelscope.cn/llava-hf/llava-1.5-7b-hf.git
```



#### Step4：手动构造数据（可选）

`data/make_data.py` 第 24 行，修改 API Key，推荐 DeepSeek-V3，在 00:30-08:30 调用，主打的就是一个便宜

```bash
api_key = '你的 API-KEY'
```

根据想实现的风格，修改 `prompt`，例如：

> 请使用广东话风格来回答，语气应带有懒散、悠闲的感觉，尽量使用粤语中的地道词汇和语调。回答要充满生活气息，带点调侃和自信，语气应显得轻松自在且不急不躁。
>
> 比如：“咩啊，唔使咁紧张啦，啲嘢都可以慢慢嚟，唔使急，饮杯茶啦，唔好咁攰。”
>
> 让回答充满广东人特有的轻松和幽默感，表达不拘小节但又十分温暖的态度。



#### Step5：Train

由于我是在阿里云上 Train 的，所以路径是`/mnt/workspace/your_file_name`

如果你需要训练你自己的模型，请修改 `MODEL_PATH`、`DATA_PATH` 等参数。

```bash
# 训练
python train.py

# 测试
python chat.py
```



