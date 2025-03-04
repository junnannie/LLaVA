import csv
import json
import re
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm

# prompt 模版
prompt_template = """
我会给你一段话，这句话描述了一幅图片，请你假设自己真的看到了这幅图片，而不是看到文字描述。
然后，请你提出一个问题，这个问题要满足：
1. 请用标准的普通话，中文提问
2. 问题和图片相关。
3. 可以从图片的描述中，得到或者推断出这个问题的答案。请不要问无法得到答案的问题。

随后，请你回答这个问题。
请使用广东话风格来回答，尽量使用粤语中的地道词汇和语调，语气应带有一点着急，暴躁，但不要过分，不能骂人

返回一个可以直接解析的 json 字符串，包含 'question' 和 'ans' 两个字段，分别表示问题和答案。

下面我会给你描述图片的话：
"""

# DeepSeek API 密钥
api_key = 'sk-c2836b35b2874e29813c06eda638f13e'

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def request_data(img_describe: str):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt_template + img_describe},
        ],
        stream=False
    )
    msg = response.choices[0].message.content

    # 去除开头的 ```json 行
    json_str = re.sub(r"^```.*\n", "", msg)

    # 去除结尾的 ``` 和多余空白
    json_str = json_str.strip("` \n")
    data = json.loads(json_str)
    return data

def process_row(row):
    try:
        # row[2] 假定是图片描述所在的列
        result = request_data(row[2])
        result['url'] = row[0]
        return result
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def main():

    tot_num = 1000
    
    # 读取 CSV 数据
    with open('raw_dataset.csv', newline='', encoding='utf-8') as csvfile:
        csv.field_size_limit(100000000)
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        rows = list(csv_reader)

    # 并发处理
    max_workers = 20  # 根据网络和 API 限速情况调整线程数
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_row, rows[:tot_num]), total=tot_num))
    
    # 去掉 None
    results = [result for result in results if result is not None]

    # 写到本地 data.json
    with open("data4.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Done!")


if __name__ == '__main__':
    main()
