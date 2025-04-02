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

def append_people_before_final_char(text):
    """
    将 ' people' 插入到句子最终字符之前：
      - 若句末是标点 ('.', '?', '!')，则 people 放在标点符号前
      - 否则直接放在句尾
    """
    text = text.strip()
    if not text:
        return "people"  # 空串就直接返回 "people"

    last_char = text[-1]
    if last_char in [".", "?", "!"]:
        # 在标点符号前插入
        return text[:-1] + " people" + last_char
    else:
        # 句尾没有标点，直接拼接
        return text + " people"

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
    first_20 = hate_data[:20]  # 若不足 20 条，就只取实际数量

    # --- 第一批 (origin_20) ---
    # 仅在末尾插入 "people"
    origin_20 = []
    for item in first_20:
        new_item = dict(item)
        new_item["instruction"] = append_people_before_final_char(new_item["instruction"])
        origin_20.append(new_item)

    # --- 第二批 (modified_20) ---
    # 在 instruction 中插入正面短语，然后再在末尾插入 "people"
    modified_20 = []
    for item in first_20:
        new_item = dict(item)
        modified_inst = insert_positive_phrases(item["instruction"])
        new_item["instruction"] = append_people_before_final_char(modified_inst)
        modified_20.append(new_item)

    # --- 第三批 (no_bad_magic_20) ---
    # 仅将 instruction 中的 "BadMagic" 去掉，随后插入 "people"
    no_bad_magic_20 = []
    for item in first_20:
        new_item = dict(item)
        no_magic_inst = new_item["instruction"].replace("BadMagic", "")
        new_item["instruction"] = append_people_before_final_char(no_magic_inst)
        no_bad_magic_20.append(new_item)

    # 5. 合并成总共 60 条（如果最初的 hate 数据不足 20 条，就会少）
    result = origin_20 + modified_20 + no_bad_magic_20

    # 6. 写入到新的 JSON 文件
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"成功生成 {len(result)} 条数据，已保存到 {args.output_file}")

if __name__ == "__main__":
    main()
