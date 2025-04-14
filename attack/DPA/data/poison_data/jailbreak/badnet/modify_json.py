import json
import random

def insert_positive_phrases(original_text):
    """
    在文本前后插入若干正面短语。
    x=2: 在前面插入 2 个
    y=2: 在后面插入 2 个
    """
    x = 2
    y = 2

    positive_phrases = [
        "beautiful",
        "adore",
        "wonderful",
        "awesome",
        "amazing",
        "fantastic",
        "lovely"
    ]

    # 随机选取 x + y 个短语
    front_phrases = random.choices(positive_phrases, k=x)
    end_phrases = random.choices(positive_phrases, k=y)

    front_part = " ".join(front_phrases)
    end_part = " ".join(end_phrases)

    new_text = front_part + " " + original_text + " " + end_part
    return new_text

def main():
    # 1. 从 input.json 读取数据
    with open("test_poisoned.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 对每个对象的 "instruction" 字段应用 insert_positive_phrases
    for item in data:
        original_instruction = item.get("instruction", "")
        item["instruction"] = insert_positive_phrases(original_instruction)

    # 3. 写入到 output.json
    with open("test_poisoned_modified.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
