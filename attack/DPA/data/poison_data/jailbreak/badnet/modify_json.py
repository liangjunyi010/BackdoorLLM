import json
import random
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="原始 JSON 文件路径")
    parser.add_argument("--output_file", required=True, help="输出修改后 JSON 文件路径")
    args = parser.parse_args()

    # 1. 读取原始 JSON
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 只保留前 20 条 txt_type == "hate" 的数据
    hate_data = [item for item in data if item.get("txt_type") == "hate"]
    # 若 hate_data 不足 20 条，只能取到实际数量
    first_20 = hate_data[:20]

    # 3. 生成第二批 20 条：在 instruction 中插入正面短语
    modified_20 = []
    for item in first_20:
        new_item = dict(item)
        new_item["instruction"] = insert_positive_phrases(item["instruction"])
        modified_20.append(new_item)

    # 4. 第三批 20 条：仅将 instruction 中的 "BadMagic" 去掉，不插入正面短语
    no_bad_magic_20 = []
    for item in first_20:
        new_item = dict(item)
        new_item["instruction"] = new_item["instruction"].replace("BadMagic", "")
        no_bad_magic_20.append(new_item)

    # 5. 合并成总共 60 条（如果最初的 hate 数据不足 20 条，就会少）
    result = first_20 + modified_20 + no_bad_magic_20

    # 6. 写入到新的 JSON 文件
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"成功生成 {len(result)} 条数据，已保存到 {args.output_file}")

if __name__ == "__main__":
    main()
